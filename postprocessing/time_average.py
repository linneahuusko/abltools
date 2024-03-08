from postcipes.nek_channel_flow_nc import NekChannelFlowNc
from timeit import default_timer as timer
import sys

time0 = timer()

casename = sys.argv[1]
N = sys.argv[2]

c = NekChannelFlowNc("./output", casename, N, 10000, None, 8, 1e-10)

c.save(f"data/{casename}.nc")
c.save_z_i("data/z_i.nc")

print(timer() - time0)
