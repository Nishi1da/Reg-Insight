"""Groq LLM Client - Fast inference via Groq API"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Optional, Any
import time
import logging
import yaml
import os
from dataclasses import dataclass
from datetime import datetime

from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response"""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'model': self.model,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'latency_ms': self.latency_ms,
            'timestamp': self.timestamp
        }


class GroqLLMClient:
    """
    Groq API client for fast LLM inference

    Features:
    - Simple API calls (no tunnels, no Colab)
    - Exponential backoff retry
    - Token usage tracking
    - Daily usage monitoring
    - Health check
    """

    # Free tier limits
    FREE_TIER_RPM = 30         # Requests per minute
    FREE_TIER_RPD = 14400      # Requests per day

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 500,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        # Load API key from parameter, env var, or config file
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or self._load_key_from_config()
        )

        if not self.api_key:
            raise ValueError(
                "Groq API key not found.\n"
                "Options:\n"
                "1. Set GROQ_API_KEY environment variable\n"
                "2. Add to config/groq_config.yaml\n"
                "3. Pass directly: GroqLLMClient(api_key='gsk_...')\n"
                "Get free key at: console.groq.com"
            )

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
            'retry_count': 0,
            'daily_requests': 0
        }

        logger.info(f"GroqLLMClient initialized: {model}")

    def _load_key_from_config(self) -> Optional[str]:
        """Load API key from config file"""
        config_path = Path("config/groq_config.yaml")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                key = config.get('groq', {}).get('api_key', '')
                if key and key != 'gsk_your_key_here':
                    return key
            except Exception:
                pass
        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text using Groq API

        Args:
            prompt: User prompt text
            system_prompt: Optional system instructions
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            LLMResponse with text and metadata
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        self.stats['daily_requests'] += 1

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens
                )

                elapsed = (time.time() - start_time) * 1000

                # Extract response data
                text = response.choices[0].message.content
                usage = response.usage

                # Update stats
                self.stats['successful_requests'] += 1
                self.stats['total_tokens'] += usage.total_tokens
                self.stats['total_latency_ms'] += elapsed
                self.stats['avg_latency_ms'] = (
                    self.stats['total_latency_ms']
                    / self.stats['successful_requests']
                )

                return LLMResponse(
                    text=text,
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    latency_ms=elapsed,
                    timestamp=datetime.now().isoformat()
                )

            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting
                if 'rate limit' in error_str or '429' in error_str:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limited. Waiting {wait}s... "
                        f"({attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                    self.stats['retry_count'] += 1
                    continue

                # Handle API errors
                elif attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Error: {e}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    self.stats['retry_count'] += 1
                    continue

                else:
                    self.stats['failed_requests'] += 1
                    logger.error(f"Generation failed after retries: {e}")
                    raise

        self.stats['failed_requests'] += 1
        raise RuntimeError("Max retries exceeded")

    def health_check(self) -> Dict[str, Any]:
        """
        Verify API key and connection work

        Returns:
            Dict with status and model info
        """
        try:
            test_response = self.generate(
                prompt='Say only: {"status": "ok"}',
                max_tokens=20
            )

            return {
                'status': 'healthy',
                'model': self.model,
                'response_preview': test_response.text[:50],
                'latency_ms': test_response.latency_ms,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }

    def check_daily_usage(self) -> Dict:
        """Check daily usage against free tier limit"""
        used = self.stats['daily_requests']
        remaining = self.FREE_TIER_RPD - used
        pct = used / self.FREE_TIER_RPD * 100

        return {
            'used_today': used,
            'remaining_today': remaining,
            'daily_limit': self.FREE_TIER_RPD,
            'usage_percent': round(pct, 1),
            'status': (
                'ok' if pct < 80
                else 'warning' if pct < 95
                else 'critical'
            )
        }

    def get_stats(self) -> Dict:
        """Get client statistics"""
        total = self.stats['total_requests']
        success_rate = (
            self.stats['successful_requests'] / total * 100
            if total > 0 else 0
        )

        return {
            **self.stats,
            'success_rate_percent': round(success_rate, 2),
            'model': self.model,
            'daily_usage': self.check_daily_usage()
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print(" Groq LLM Client Test")
    print("=" * 60)

    print("\n1. Initializing Groq client...")
    try:
        client = GroqLLMClient(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=200
        )
        print(f"    Client initialized")
        print(f"   Model: {client.model}")
    except ValueError as e:
        print(f"    {e}")
        print("\n   Add your API key to config/groq_config.yaml")
        print("   Then run this test again")
        exit()

    print("\n2. Health check...")
    health = client.health_check()
    print(f"   Status: {health['status']}")

    if health['status'] == 'healthy':
        print(f"   Latency: {health['latency_ms']:.0f}ms")
        print(f"   Response: {health['response_preview']}")

        print("\n3. Testing compliance explanation generation...")
        test_prompt = """Analyze this gap:

REGULATION: Organizations must implement multi-factor authentication.
POLICY: All users must use strong passwords of 12+ characters.
SCORE: 0.28/1.0

Respond with JSON only:
{
    "summary": "brief assessment",
    "recommendation": "specific action",
    "risk_level": "high",
    "key_differences": ["MFA missing", "Only password covered"],
    "confidence": "high"
}"""

        response = client.generate(
            prompt=test_prompt,
            system_prompt="You are a compliance analyst. Respond with JSON only."
        )

        print(f"    Generated in {response.latency_ms:.0f}ms")
        print(f"   Tokens used: {response.total_tokens}")
        print(f"   Preview: {response.text[:150]}")

        print("\n4. Daily usage check...")
        usage = client.check_daily_usage()
        print(f"   Used today: {usage['used_today']}/{usage['daily_limit']}")
        print(f"   Remaining: {usage['remaining_today']}")
        print(f"   Status: {usage['status']}")

    print("\n5. Client statistics:")
    stats = client.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Success rate: {stats['success_rate_percent']}%")
    print(f"   Avg latency: {stats['avg_latency_ms']:.0f}ms")


    print("\n" + "=" * 60)
    print("=" * 60)