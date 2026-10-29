#define main generated_benchmark_main
#include "../proposal/proposal_bench.cu"
#undef main
#include <fstream>
#include <limits>

static matrix_csr read_real_csr(const char *path) {
  std::ifstream in(path);std::string magic;int n,m,nnz;
  if (!(in>>magic>>n>>m>>nnz) || magic!="PCSR1" || n<=0 || m<=0 || nnz<=0)
    throw std::runtime_error("Invalid CSR header");
  if (n>1048576 || m>1048576 || nnz>67108864) throw std::runtime_error("CSR exceeds input limits");
  matrix_csr csr;csr.columns=m;csr.offsets.resize(n+1);csr.indices.resize(nnz);csr.values.resize(nnz);
  for (int &v:csr.offsets) if (!(in>>v)) throw std::runtime_error("Missing offsets");
  for (int &v:csr.indices) if (!(in>>v) || v<0 || v>=m) throw std::runtime_error("Invalid column");
  for (float &v:csr.values) if (!(in>>v) || !std::isfinite(v)) throw std::runtime_error("Invalid value");
  if (csr.offsets.front()!=0 || csr.offsets.back()!=nnz ||
      !std::is_sorted(csr.offsets.begin(),csr.offsets.end()))throw std::runtime_error("Invalid row offsets");
  return csr;
}

__global__ void real_bfs_pull(int n,const int *offsets,const int *parents,
    const int *frontier,int *next_frontier,int *distance,int level,int stride) {
  int v=blockIdx.x*blockDim.x+threadIdx.x;
  if(v>=n)return;
  next_frontier[v*stride]=0;
  if(distance[v]>=0)return;
  for(int e=offsets[v];e<offsets[v+1];++e) {
    if(frontier[parents[e]*stride]) {
      distance[v]=level+1;next_frontier[v*stride]=1;break;
    }
  }
}

static void real_bfs(const matrix_csr &csr,int stride) {
  int n=static_cast<int>(csr.offsets.size())-1;
  if(csr.columns!=n)throw std::runtime_error("Graph must be square");
  graph_csr g;g.offsets=csr.offsets;g.columns=csr.indices;
  auto expected=cpu_bfs(g);std::vector<int> actual(n,-1);actual[0]=0;
  std::vector<int> offsets(n+1,0),parents(g.columns.size());
  for(int u:g.columns)++offsets[u+1];
  for(int i=0;i<n;++i)offsets[i+1]+=offsets[i];
  auto cursor=offsets;
  for(int v=0;v<n;++v)for(int e=g.offsets[v];e<g.offsets[v+1];++e)parents[cursor[g.columns[e]]++]=v;
  std::vector<int> first(n*stride,0),next(n*stride,0);first[0]=1;
  device_buffer<int> d_offsets(offsets.size()),d_parents(parents.size()),distance(n),frontier(first.size()),next_frontier(next.size());
  d_offsets.upload(offsets);d_parents.upload(parents);distance.upload(actual);frontier.upload(first);next_frontier.upload(next);
  int launches=0;
  for(int level=0;level<n;++level) {
    real_bfs_pull<<<(n+511)/512,512>>>(n,d_offsets.data,d_parents.data,frontier.data,next_frontier.data,distance.data,level,stride);
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());++launches;
    next_frontier.download(next);bool changed=false;
    for(int v=0;v<n;++v)changed|=next[v*stride]!=0;
    if(!changed)break;
    std::swap(frontier.data,next_frontier.data);
    if(level==n-1)throw std::runtime_error("BFS failed to converge");
  }
  distance.download(actual);int reached=0,max_distance=0;unsigned long long checksum=0;
  for(int v=0;v<n;++v) {
    if(actual[v]!=expected[v])throw std::runtime_error("Real BFS mismatch at vertex "+std::to_string(v));
    if(actual[v]>=0){++reached;max_distance=std::max(max_distance,actual[v]);checksum+=(v+1ULL)*(actual[v]+1ULL);}
  }
  std::printf("REAL_PASS workload=bfs dataset=ca-GrQc stride=%d n=%d nnz=%zu reachable=%d max_distance=%d reference_checksum=%llu logical_vector_bytes=%zu physical_vector_bytes=%zu kernels=%d\n",
      stride,n,g.columns.size(),reached,max_distance,checksum,n*sizeof(int),first.size()*sizeof(int),launches);
}

static void real_spmv(const matrix_csr &csr,int stride) {
  int n=static_cast<int>(csr.offsets.size())-1;
  std::vector<float> logical(csr.columns),physical(csr.columns*stride,std::numeric_limits<float>::quiet_NaN()),actual(n);
  for(unsigned i=0;i<logical.size();++i) {
    logical[i]=(static_cast<int>(((i*1664525u+1013904223u)>>16)%17)-8)*0.25f;
    physical[i*stride]=logical[i];
  }
  auto mapped=csr.indices;for(int &col:mapped)col*=stride;
  std::vector<double> expected(n,0),absolute_sum(n,0),bound(n,0);
  double checksum=0;const double eps=std::numeric_limits<float>::epsilon();
  for(int row=0;row<n;++row) {
    for(int e=csr.offsets[row];e<csr.offsets[row+1];++e) {
      double product=static_cast<double>(csr.values[e])*logical[csr.indices[e]];
      expected[row]+=product;absolute_sum[row]+=std::fabs(product);
    }
    int k=csr.offsets[row+1]-csr.offsets[row];
    double gamma=2.0*k*eps/(1.0-2.0*k*eps);
    bound[row]=gamma*absolute_sum[row]+1e-6;
    checksum+=expected[row]*(1+row%31);
  }
  device_buffer<int> offsets(csr.offsets.size()),indices(mapped.size());
  device_buffer<float> values(csr.values.size()),x(physical.size()),y(n);
  offsets.upload(csr.offsets);indices.upload(mapped);values.upload(csr.values);x.upload(physical);
  spmv_csr<<<(n+511)/512,512>>>(n,offsets.data,indices.data,values.data,x.data,y.data);
  CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());y.download(actual);
  double max_error=0,max_ratio=0;
  for(int row=0;row<n;++row) {
    double error=std::fabs(actual[row]-expected[row]);
    if(!std::isfinite(actual[row]) || error>bound[row]) {
      std::printf("REAL_ERROR row=%d actual=%.9g expected=%.17g error=%.9g bound=%.9g\n",row,actual[row],expected[row],error,bound[row]);
      throw std::runtime_error("Real SpMV validation failed");
    }
    max_error=std::max(max_error,error);max_ratio=std::max(max_ratio,error/bound[row]);
  }
  std::printf("REAL_PASS workload=spmv dataset=bcsstk17 stride=%d n=%d nnz=%zu reference_checksum=%.17g max_error=%.9g max_bound_ratio=%.9g logical_vector_bytes=%zu physical_vector_bytes=%zu kernels=1\n",
      stride,n,csr.indices.size(),checksum,max_error,max_ratio,logical.size()*sizeof(float),physical.size()*sizeof(float));
}

int main(int argc,char **argv) {
  try {
    if(argc!=4)throw std::runtime_error("Usage: real_bench bfs|spmv input.csr stride(1|32)");
    int stride=std::atoi(argv[3]);if(stride!=1 && stride!=32)throw std::runtime_error("Unsupported stride");
    auto csr=read_real_csr(argv[2]);
    if(!std::strcmp(argv[1],"bfs"))real_bfs(csr,stride);
    else if(!std::strcmp(argv[1],"spmv"))real_spmv(csr,stride);
    else throw std::runtime_error("Unknown workload");
    return 0;
  }catch(const std::exception &e){std::fprintf(stderr,"REAL_FAIL %s\n",e.what());return 1;}
}
