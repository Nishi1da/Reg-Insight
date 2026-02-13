# Chunking Strategy Documentation

## Test Environment
- **Chunk Size Tested:** [256, 512,1024]
- **Overlap Tested:** [0, 50, 100, 200]
- **PDFs Used:**AML regulation,AML policy,data protection policy,data protection regulation

## Key Findings

### What Worked Well
- 512 chunk size preserved paragraph boundaries
- 50 overlap maintained context between chunks

### What Didn't Work
- 256 size cut sentences in half
- 0 overlap lost context at chunk boundaries

## Selected Configuration

- **Chunk Size:** 512
- **Overlap:** 50
- **Reason:**
- Overlap of 50 creates expected word repetition between chunks (preserves context)
- No mid-sentence cuts at 512
