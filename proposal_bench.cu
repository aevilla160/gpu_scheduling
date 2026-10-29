#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CUDA(call) check((call), #call)

template <class T> struct device_buffer {
  T *data = nullptr;
  explicit device_buffer(size_t count) {
    CUDA(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));
  }
  ~device_buffer() { if (data) cudaFree(data); }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
  void upload(const std::vector<T> &host) {
    CUDA(cudaMemcpy(data, host.data(), host.size()*sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(std::vector<T> &host) {
    CUDA(cudaMemcpy(host.data(), data, host.size()*sizeof(T), cudaMemcpyDeviceToHost));
  }
};

struct graph_csr {
  std::vector<int> offsets, columns;
};

static graph_csr make_graph(int n) {
  graph_csr g;
  g.offsets.push_back(0);
  const int reachable = 3*n/4;
  for (int v=0; v<n; ++v) {
    std::vector<int> targets;
    if (v < reachable) {
      // A rooted binary tree guarantees reachability of this component.
      if (2*v+1 < reachable) targets.push_back(2*v+1);
      if (2*v+2 < reachable) targets.push_back(2*v+2);
      if (v) targets.push_back(v-1);
      // Nonuniform degree, cycles and competing discovery paths.
      int degree = v%17 == 0 ? 24 : v%9;
      for (int k=0; k<degree; ++k)
        targets.push_back((v*17+k*13+7)%reachable);
    } else {
      // A separate component must remain unreachable from root zero.
      targets.push_back(reachable+(v-reachable+1)%(n-reachable));
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    g.columns.insert(g.columns.end(), targets.begin(), targets.end());
    g.offsets.push_back(static_cast<int>(g.columns.size()));
  }
  return g;
}

static std::vector<int> cpu_bfs(const graph_csr &g) {
  const int n=static_cast<int>(g.offsets.size())-1;
  std::vector<int> distance(n,-1);
  std::queue<int> frontier;
  distance[0]=0; frontier.push(0);
  while (!frontier.empty()) {
    int v=frontier.front(); frontier.pop();
    for (int e=g.offsets[v]; e<g.offsets[v+1]; ++e) {
      int u=g.columns[e];
      if (distance[u] == -1) {
        distance[u]=distance[v]+1;
        frontier.push(u);
      }
    }
  }
  return distance;
}

__global__ void bfs_step(int n, const int *offsets, const int *parents,
                         const int *frontier, int *next_frontier,
                         int *distance, int level) {
  int v=blockIdx.x*blockDim.x+threadIdx.x;
  if (v>=n) return;
  next_frontier[v]=0;
  if (distance[v]>=0) return;
  for (int e=offsets[v];e<offsets[v+1];++e) {
    if (frontier[parents[e]]) {
      distance[v]=level+1;
      next_frontier[v]=1;
      break;
    }
  }
}

static void run_bfs(int n) {
  graph_csr g=make_graph(n);
  std::vector<int> expected=cpu_bfs(g), actual(n,-1), changed(1,0);
  actual[0]=0;
  // Transpose outgoing CSR for destination-owned pull traversal.
  std::vector<int> incoming_offsets(n+1,0), parents(g.columns.size());
  for (int u:g.columns) ++incoming_offsets[u+1];
  for (int v=0;v<n;++v) incoming_offsets[v+1]+=incoming_offsets[v];
  std::vector<int> cursor=incoming_offsets;
  for (int v=0;v<n;++v)
    for (int e=g.offsets[v];e<g.offsets[v+1];++e) parents[cursor[g.columns[e]]++]=v;
  std::vector<int> first_frontier(n,0), next_mask(n,0);
  first_frontier[0]=1;
  device_buffer<int> offsets(incoming_offsets.size()), columns(parents.size());
  device_buffer<int> distance(n), frontier(n), next_frontier(n);
  offsets.upload(incoming_offsets); columns.upload(parents); distance.upload(actual);
  frontier.upload(first_frontier);
  int launches=0;
  for (int level=0; level<n; ++level) {
    bfs_step<<<(n+511)/512,512>>>(n,offsets.data,columns.data,frontier.data,
        next_frontier.data,distance.data,level);
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    ++launches; next_frontier.download(next_mask);
    changed[0]=std::any_of(next_mask.begin(),next_mask.end(),[](int x){return x!=0;});
    if (std::getenv("BENCH_DEBUG")) {
      std::vector<int> snapshot(n);
      distance.download(snapshot);
      int mismatches=0;
      for (int v=0; v<n; ++v) {
        int want=expected[v]>=0 && expected[v]<=level+1 ? expected[v] : -1;
        if (snapshot[v]!=want) {
          if (mismatches<8) std::printf("BFS_DEBUG level=%d vertex=%d actual=%d expected=%d\n",level,v,snapshot[v],want);
          ++mismatches;
        }
      }
      std::printf("BFS_LEVEL level=%d changed=%d mismatches=%d\n",level,changed[0],mismatches);
      std::vector<int> next_snapshot(n);
      next_frontier.download(next_snapshot);
      int frontier_mismatches=0;
      for (int v=0; v<n; ++v) {
        int want=expected[v]==level+1;
        if (next_snapshot[v]!=want) {
          if (frontier_mismatches<8) std::printf("FRONTIER_DEBUG level=%d vertex=%d actual=%d expected=%d\n",level,v,next_snapshot[v],want);
          ++frontier_mismatches;
        }
      }
      std::printf("FRONTIER_LEVEL level=%d mismatches=%d\n",level,frontier_mismatches);
    }
    if (!changed[0]) break;
    std::swap(frontier.data,next_frontier.data);
    if (level==n-1) throw std::runtime_error("BFS failed to converge");
  }
  distance.download(actual);
  int reached=0, max_distance=0;
  for (int i=0; i<n; ++i) {
    if (actual[i]!=expected[i])
      throw std::runtime_error("BFS distance mismatch at vertex "+std::to_string(i)+
          " actual="+std::to_string(actual[i])+" expected="+std::to_string(expected[i]));
    if (actual[i]>=0) { ++reached; max_distance=std::max(max_distance,actual[i]); }
  }
  if (reached!=3*n/4) throw std::runtime_error("BFS component coverage failed");
  std::printf("BENCHMARK_PASS workload=bfs algorithm=pull n=%d edges=%zu reachable=%d unreachable=%d max_distance=%d kernels=%d\n",
              n,g.columns.size(),reached,n-reached,max_distance,launches);
}

struct matrix_csr {
  int columns;
  std::vector<int> offsets, indices;
  std::vector<float> values;
};

static matrix_csr make_matrix(int rows) {
  matrix_csr m;
  m.columns=1;
  while (m.columns<2*rows) m.columns*=2;
  m.offsets.push_back(0);
  for (int r=0; r<rows; ++r) {
    int count=r%13==0 ? 0 : r%29==0 ? 48 : 1+r%11;
    for (int j=0; j<count; ++j) {
      m.indices.push_back((r*13+j*17)%m.columns);
      m.values.push_back(((r+3*j)%9-4)*0.125f);
    }
    m.offsets.push_back(static_cast<int>(m.indices.size()));
  }
  return m;
}

__global__ void spmv_csr(int rows, const int *offsets, const int *indices,
                         const float *values, const float *x, float *y) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if (row>=rows) return;
  float sum=0;
  for (int e=offsets[row]; e<offsets[row+1]; ++e)
    sum+=values[e]*x[indices[e]];
  y[row]=sum;
}

static void run_spmv(int rows) {
  matrix_csr m=make_matrix(rows);
  std::vector<float> x(m.columns), actual(rows,-12345.0f);
  for (int i=0; i<m.columns; ++i) x[i]=(i%17-8)*0.25f;
  std::vector<double> expected(rows,0);
  int empty_rows=0;
  for (int r=0; r<rows; ++r) {
    if (m.offsets[r]==m.offsets[r+1]) ++empty_rows;
    for (int e=m.offsets[r]; e<m.offsets[r+1]; ++e)
      expected[r]+=static_cast<double>(m.values[e])*x[m.indices[e]];
  }
  device_buffer<int> offsets(m.offsets.size()), indices(m.indices.size());
  device_buffer<float> values(m.values.size()), input(x.size()), output(rows);
  offsets.upload(m.offsets); indices.upload(m.indices); values.upload(m.values);
  input.upload(x); output.upload(actual);
  spmv_csr<<<(rows+511)/512,512>>>(rows,offsets.data,indices.data,values.data,input.data,output.data);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize()); output.download(actual);
  double max_error=0;
  for (int r=0; r<rows; ++r) {
    double error=std::fabs(actual[r]-expected[r]);
    if (!std::isfinite(actual[r]) || error>1e-5*(1+std::fabs(expected[r])))
      throw std::runtime_error("SpMV mismatch at row "+std::to_string(r));
    max_error=std::max(max_error,error);
  }
  std::printf("BENCHMARK_PASS workload=spmv n=%d columns=%d nnz=%zu empty_rows=%d max_error=%.9g kernels=1\n",
              rows,m.columns,m.values.size(),empty_rows,max_error);
}

__host__ __device__ static float4 compute_reference(unsigned i, int rounds) {
  float v=0.25f+(i%31)*0.03125f;
  float4 a=make_float4(v,v+0.125f,v-0.0625f,v+0.25f);
  for (int j=0; j<rounds; ++j) {
    // Exact binary fractions keep the control independent of backend
    // differences in multiply-add rounding. Retain four independent chains.
    float term=v+j*0.03125f;
    a.x=fmaf(term,0.5f,a.x);
    a.y=fmaf(term,0.25f,a.y);
    a.z=fmaf(term,0.125f,a.z);
    a.w=fmaf(term,0.0625f,a.w);
  }
  return a;
}

__global__ void compute_control(int n, int rounds, float4 *output) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) output[i]=compute_reference(i,rounds);
}

static void run_compute(int n) {
  const int rounds=64;
  std::vector<float4> actual(n);
  device_buffer<float4> output(n);
  compute_control<<<(n+511)/512,512>>>(n,rounds,output.data);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize()); output.download(actual);
  double max_error=0;
  for (int i=0; i<n; ++i) {
    float4 expected=compute_reference(i,rounds);
    float a[]={actual[i].x,actual[i].y,actual[i].z,actual[i].w};
    float b[]={expected.x,expected.y,expected.z,expected.w};
    for (int k=0; k<4; ++k) {
      double error=std::fabs(static_cast<double>(a[k])-b[k]);
      if (!std::isfinite(a[k]) || error>1e-6*(1+std::fabs(b[k])))
      {
        std::printf("COMPUTE_DEBUG thread=%d component=%d actual=%.9g expected=%.9g error=%.9g\n",i,k,a[k],b[k],error);
        throw std::runtime_error("Compute mismatch at thread "+std::to_string(i));
      }
      max_error=std::max(max_error,error);
    }
  }
  std::printf("BENCHMARK_PASS workload=compute n=%d rounds=%d fma_chains=4 max_error=%.9g kernels=1\n",n,rounds,max_error);
}

int main(int argc,char **argv) {
  try {
    if (argc!=3) throw std::runtime_error("Usage: proposal_bench bfs|spmv|compute size");
    char *end=nullptr;
    long size=std::strtol(argv[2],&end,10);
    if (!end || *end || size<8 || size>65536) throw std::runtime_error("Size must be 8..65536");
    if (!std::strcmp(argv[1],"bfs")) run_bfs(size);
    else if (!std::strcmp(argv[1],"spmv")) run_spmv(size);
    else if (!std::strcmp(argv[1],"compute")) run_compute(size);
    else throw std::runtime_error("Unknown workload");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr,"BENCHMARK_FAIL %s\n",e.what()); return 1;
  }
}
