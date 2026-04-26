import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    PID = tl.program_id(axis=0) #this is per-block, it traverses in the grid along the first dimension.
    block_start = PID * BLOCK_SIZE #so when PID = 0 and BLOCK_SIZE=1024, it's 0, PID=1 block_start=1024 - basically just getting the start index of the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE) #PID=0 offsets=Tensor[0,..,1023]), PID=1 offsets=Tensor[1024...2047]
    mask = offsets < n_elements #say that we have grid=2, block_size=1024, n_elements=1026, the rest 1022 block will be running on nothing, mask is here to save this, it ensures only valid indices are processed. Mask is per-thread, False masks skip memory access for the certain thread but will still execute the kernal.

    # 'load' here grabs data from DRAM/VRAM/HBM to SRAM/on-chip memory
    x = tl.load(x_ptr + offsets, mask = mask, other = None) # other = None means that for masked out indices, no fallback value will be provided as Triton will leave them undefined, and load on those indices will be skipped
    y = tl.load(y_ptr + offsets, mask = mask, other = None)

    output = x + y

    # write data back to DRAM. output_ptr+offsets gives the index of where to write, and output is the real output :) flag shall still get passed here to be safe
    tl.store(output_ptr + offsets, output, mask = mask)
    
    

def add(x,y):
    #create z
    output = torch.empty_like(x)
    #check if tensors are in the same device
    assert x.device == DEVICE and y.device == DEVICE

    #launch grid, we want it to be a tuple. It defines how many different programs/instances of kernels will get called.
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # ceilling division, cdiv(m,n)=m + (n-1) // n
    # there's a -1 to avoid running whole unused block in case n is a perfect divident - say 96+32/32->4 when we actually only needs 3 blocks
    
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )

    return output
    

#absolute tol and relative tol
def test_add_kernel(size, atol=1e-3, rtol=1e-3, device = DEVICE):
    # test data
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    #run triton kernal & pytorch equivalent
    z_tri=add(x, y)
    z_ref=x+y
    #compare
    torch.testing.assert_close(z_tri, z_ref, atol = atol, rtol = rtol)
    print("passed")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], #actual x axis (?)
        x_vals=[2**i for i in range(12,28,1)],
        x_log = True, #logarithmic scale
        line_arg = 'provider',
        line_vals = ['Triton','Torch'],
        line_names = ['Triton', 'Torch'],
        styles = [('blue','-'), ('green','-')],
        ylabel = 'GB/s',
        plot_name = 'vector-add-performance',
        args={},
    )
)

def benchmark(size,provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5,0.05,0.95] #median, 5% slowest, and 5% fastest
    if provider == 'Torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+y, quantiles = quantiles)
    if provider == 'Triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x,y), quantiles = quantiles)

    # millisec of the ops = rounds of memory ops * total num of elements * size of each elements * bytes -> gigabytes * (ms to sec)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3) # cuz we did 3 memory ops, 1e-9 converts raw bytes to gigabytes

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True)