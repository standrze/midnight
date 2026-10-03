// Diagnostic only: supported fixed-capacity masked attention, not model integration.
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/midnight_replay.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include <iostream>
#include <vector>
#include <limits>
#include <stdexcept>
using namespace mlx::core;
static void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static std::vector<unsigned char> read(const array& a) {
  std::vector<unsigned char> r(a.nbytes());
  check(cudaMemcpy(r.data(), gpu_ptr<void>(a), a.nbytes(), cudaMemcpyDeviceToHost));
  return r;
}
int main() {
  try {
    auto s=default_stream(Device::gpu);
    auto& enc=cu::get_command_encoder(s);
    constexpr int capacity=2048;
    auto q=astype(reshape(sin(multiply(arange(896,float32,s),array(.013f),s),s),{1,14,1,64},s),float16,s);
    auto k=astype(reshape(cos(multiply(arange(2*capacity*64,float32,s),array(.007f),s),s),{1,2,capacity,64},s),float16,s);
    auto v=astype(reshape(multiply(sin(multiply(arange(2*capacity*64,float32,s),array(.019f),s),s),array(.25f),s),{1,2,capacity,64},s),float16,s);
    auto q2=negative(q,s); auto k2=negative(k,s); auto v2=negative(v,s);
    auto mask=zeros({1,1,1,capacity},float16,s);
    eval({q,k,v,q2,k2,v2,mask}); synchronize(s);
    auto attention=[&] {return fast::scaled_dot_product_attention(q,k,v,.125f,"",mask,std::nullopt,s);};
    auto bindings=[&] {return std::vector<array>{q,k,v,mask};};
    auto warm=attention(); eval(warm); synchronize(s);
    cu::ReplaySession session(enc);
    array output=q;
    session.record(1,bindings(),[&] {output=attention();eval(output);});
    int failures=0,checks=0;
    std::vector<unsigned char> previous;
    for(int length : {1,17,255,256,257,511,512,1023,1834,2048,17,1}) {
      std::vector<float> m(capacity,-std::numeric_limits<float>::infinity());
      for(int i=0;i<length;++i)m[i]=0;
      auto next=astype(array(m.begin(),{1,1,1,capacity}),float16,s);
      eval(next);synchronize(s);
      check(cudaMemcpyAsync(gpu_ptr<void>(mask),gpu_ptr<void>(next),mask.nbytes(),cudaMemcpyDeviceToDevice,enc.stream()));
      // Mutate stable addresses at two points, exercising query and cache feedback.
      if(checks==4)check(cudaMemcpyAsync(gpu_ptr<void>(q),gpu_ptr<void>(q2),q.nbytes(),cudaMemcpyDeviceToDevice,enc.stream()));
      if(checks==8) {
        check(cudaMemcpyAsync(gpu_ptr<void>(k),gpu_ptr<void>(k2),k.nbytes(),cudaMemcpyDeviceToDevice,enc.stream()));
        check(cudaMemcpyAsync(gpu_ptr<void>(v),gpu_ptr<void>(v2),v.nbytes(),cudaMemcpyDeviceToDevice,enc.stream()));
      }
      session.replay(1,bindings());auto actual=read(output);
      auto ordinary=attention();eval(ordinary);synchronize(s);auto expected=read(ordinary);
      bool exact=actual==expected,repeat=true;
      for(int j=0;j<10;++j){session.replay(1,bindings());repeat &= read(output)==actual;}
      bool changed=previous.empty()||previous!=actual;
      failures+=!exact||!repeat||!changed;
      std::cout<<"{\"length\":"<<length<<",\"ordinary_exact\":"<<(exact?"true":"false")<<",\"ten_replays_exact\":"<<(repeat?"true":"false")<<",\"output_changed\":"<<(changed?"true":"false")<<"}"<<std::endl;
      previous=actual;++checks;
    }
    auto stats=session.statistics();session.clear();
    if(session.statistics().retained_bytes!=0)throw std::runtime_error("clear retained storage");
    std::cout<<"{\"passed\":"<<(failures==0?"true":"false")<<",\"cases\":"<<checks<<",\"retained_bytes\":"<<stats.retained_bytes<<",\"full_model\":false}"<<std::endl;
    clear_streams();return failures?1:0;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;try{clear_streams();}catch(...){}return 1;}
}
