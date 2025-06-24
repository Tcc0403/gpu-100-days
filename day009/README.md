# Day 009 (2025/6/24)
- Implemented online softmax kernel in mojo
- Wrote unit test for it

## What I learned

### Online softmax (not really new to me)

- Reference: https://arxiv.org/abs/1805.02867
- I already knew online softmax, but only did it in triton before. I was just rewriting it to get familiar with kernel developement in mojo.

## Pitfalls

You must pass all parameters defined (match the interface) in mojo when calling the mojo kernel with torch.compile! Or you will see SEGFAULT without any information again... 

### Context

I finished my custom softmax kernel. But whenever I called it from torch side, it always ran into SEGFAULT. 
### What I tried when debugging:
1. Tried the official doc about [GPU debugging](https://docs.modular.com/mojo/tools/gpu-debugging/). I couldn't figure out how to make the debugger work on python side.
    - No debugger, so just write some smoke tests then
2. Manually wrote smoke tests on both python side and mojo side. Mojo side ended peacefully, but Python side never worked.
    - I guess it's something similar to day008's bug.
    - I did set dtype when generating random tensors...
3. Decided to debug with the most silly way, commenting out code sections by sections. 
    - I thought it was invalid address at first
        - Commented out every single memory accesses, still SEGFAULT
        - Commented all kernel code, still SEGFAULT...
            - So the SEGFAULT is not due to my custom kernel
    - Commented out the entire `if target == "gpu": ...` block, still SEGFAULT
        - What is going on?
4. Started RTFM
    - https://docs.modular.com/max/api/python/torch
    - https://docs.modular.com/max/custom-ops/
    - I thought I did follow every necessary steps. 
5. Checked all available examples 
    - https://github.com/modular/modular/tree/main/examples/pytorch_custom_ops
        - [addition.py](https://github.com/modular/modular/blob/main/examples/pytorch_custom_ops/addition.py): Nothing special
        - [grayscale.py](https://github.com/modular/modular/blob/main/examples/pytorch_custom_ops/grayscale.py): That's what I did yesterday
        - [whiper.py](https://github.com/modular/modular/blob/main/examples/pytorch_custom_ops/whisper.py): Too much words (But it has the answer now I check)
    - Back to mojo-gpu-puzzles examples
        - Found another example in [p19 embedding](https://github.com/modular/mojo-gpu-puzzles/blob/main/solutions/p19/p19.py): [2 seperate lines](https://github.com/modular/mojo-gpu-puzzles/blob/aad429afa492a970db1c0ce4f6f327181a2e1b1a/solutions/p19/p19.py#L23) when calling mojo kernel. 
            - First line: Instantiate the custom op function
                - `my_ops = ops.op_name[{parameter dict}]`
            - Second line: Call the function
                - `my_ops(output, input)`
        - I did give the default parameter [in my mojo code](https://github.com/Tcc0403/gpu-100-days/blob/ce64ba62e2a84a6c4ba6b14004f0f0a809396439/pytorch_custom_ops/src/ops/softmax.mojo#L61). It should be working without passing it in pytorch, right?
            - **Nope**. Commented this single line out solved the problem
            - Lesson learned.
6. Started debugging the my incorrect kernel

### What went wrong
- It really was the same type of bugs from yesterday
- Default values for parameters don't work on pytorch side

## Takeaways
- **Always match the kernel interface on python side and mojo side**