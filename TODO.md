## TODO

1. gradient checkpoint
2. memory gradient
3. memory inference
4. YaRN

## Question

```python
# max_seq_len： 4096 * 4， original_seq_len： 4096，args.mscale： 1， args.rope_factor： 40
if args.max_seq_len > args.original_seq_len:
    mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
    self.softmax_scale = self.softmax_scale * mscale * mscale
 ```