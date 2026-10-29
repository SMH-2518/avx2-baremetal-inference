#include <bits/stdc++.h>
#include<chrono>
#include<immintrin.h>
#include<stdexcept>
class LinearArena{
    private:
    uint8_t* buffer;
    size_t capacity;
    size_t offset;
    public:
    LinearArena(size_t max_bytes):capacity(max_bytes),offset(0){
        buffer=static_cast<uint8_t*>(aligned_alloc(64,capacity));
        if(!buffer)throw std::bad_alloc();
    }
    ~LinearArena(){free(buffer);}
    template<typename T>
    T* allocate(size_t count){
        size_t request_bytes=count*sizeof(T);
        size_t aligned_offset=(offset+63)&~63;
        if(aligned_offset+request_bytes>capacity)throw std::bad_alloc();
        T* ptr=reinterpret_cast<T*>(buffer+aligned_offset);
        offset=aligned_offset+request_bytes;
        return ptr;
    }
    void reset(){offset=0;}
    size_t get_used_bytes()const{return offset;}
};
struct DenseLayer{
    float *weights;float *bias;int in_features;int out_features;
};
void skin_gemm(const float *a,const float *b,const float *bias,float *c,int n,int m,int k,bool relu){
for(int i=0;i<n;++i){
    for(int j=0;j<k;++j){
        float sum=0.0f;
        for(int x=0;x<m;++x){
            sum+=a[i*m+x]*b[x*k+j];
        }sum+=bias[j];
        if(relu&&sum<0.0f){sum=0.0f;}c[i*k+j]=sum;
    }
}
}
void rect_gemm(const float *a, const float *b,const float *bias, float *c, int n, int m, int k,bool relu){
    alignas(64) float packed_a[64*64];
    alignas(64) float packed_b[64*64];
    alignas(64) float packed_c[64*64];
    for (int si=0;si<n;si+=64){
        for(int sj=0;sj<k;sj+=64){
            for(int i=0;i<64*64;++i){
                packed_c[i]=0.0f;
            }
            for (int sx=0;sx<m;sx+=64){
                for (int i=0;i<64;++i) {
                    for (int x=0;x<64;++x) {
                        if (si+i<n&&sx+x<m) {
                            packed_a[i*64+x]=a[(si+i)*m+(sx + x)];
                        } else {
                            packed_a[i*64+x]=0.0f;
                        }
                    }
                }
                for (int x=0;x<64;++x) {
                    for (int j=0;j<64;++j) {
                        if (sx+x<m&&sj+j<k){
                            packed_b[x*64+j]=b[(sx+x)*k+(sj+j)];
                        } else {
                            packed_b[x*64+j]=0.0f;
                        }
                    }
                }
                for (int i=0;i<64;++i) {
                    for (int x=0;x<64;++x) {
                        float a_val=packed_a[i * 64 + x];
                        __m256 va=_mm256_set1_ps(a_val);                        
                        for (int j=0;j<64;j+=8) {
                        __m256 vb=_mm256_loadu_ps(&packed_b[x*64+j]);
                        __m256 vc=_mm256_loadu_ps(&packed_c[i*64+j]);
                           vc=_mm256_fmadd_ps(va,vb,vc);
                           _mm256_storeu_ps(&packed_c[i*64+j],vc);
                        }
                    }
                }
            }
            for (int i=0;i<64;++i) {
                for (int j=0;j<64;++j) {
                    if (si+i<n&&sj+j<k) {
                        float val=packed_c[i*64+j]+bias[sj+j];
                        if(relu&&val<0.0f)val=0.0f;
                        c[(si+i)*k+(sj+j)]=val;
                    }
                }
            }
        }
    }
}
void execute_layer(const float*a,const float *b,const float *bias,float *c,int n,int m,int k,bool relu){
    if(m<64||k<64){
        skin_gemm(a,b,bias,c,n,m,k,relu);
    }else {
        rect_gemm(a,b,bias,c,n,m,k,relu);
    }
}
void load_weights(const char* filepath,std::vector<DenseLayer>& layers,LinearArena& perm_arena){
    FILE* f=fopen(filepath,"rb");
    if(!f)throw std::runtime_error("Could not open weights.bin.Run the Python exporter first.");
    int num_layers;
    if(fread(&num_layers,sizeof(int),1,f)!=1)throw std::runtime_error("err");
    for(int i=0;i<num_layers;++i){
        DenseLayer layer;
        if(fread(&layer.in_features,sizeof(int),1,f)!=1)throw std::runtime_error("err");
        if(fread(&layer.out_features,sizeof(int),1,f)!=1)throw std::runtime_error("err");
        layer.weights=perm_arena.allocate<float>(layer.in_features*layer.out_features);
        layer.bias=perm_arena.allocate<float>(layer.out_features);
        size_t w_sz=layer.in_features*layer.out_features;
        if(fread(layer.weights,sizeof(float),w_sz,f)!=w_sz)throw std::runtime_error("err");
        if(fread(layer.bias,sizeof(float),layer.out_features,f)!=static_cast<size_t>(layer.out_features))throw std::runtime_error("err");
        layers.push_back(layer);
    }fclose(f);
}
void run_inference(const std::vector<DenseLayer>& layers,const float* input,float* output,LinearArena& compute_arena,int batch_size){
    int max_dim=layers[0].in_features;
    for(const auto& l:layers) if(l.out_features>max_dim)max_dim=l.out_features;
    float* buffer_A=compute_arena.allocate<float>(batch_size*max_dim);
    float* buffer_B=compute_arena.allocate<float>(batch_size*max_dim);
    const float* curr_in=input;
    float* curr_out=buffer_A;
    for(size_t i=0;i<layers.size();++i){
        bool is_last=(i==layers.size()-1);
        if(is_last)curr_out=output;
        execute_layer(curr_in,layers[i].weights,layers[i].bias,curr_out,batch_size,layers[i].in_features,layers[i].out_features,!is_last);
        curr_in=curr_out;
        curr_out=(curr_out==buffer_A)?buffer_B:buffer_A;
    }compute_arena.reset();
}
int main(){
    try{
        LinearArena perm_arena(50*1024*1024);
        LinearArena compute_arena(10*1024*1024);
        std::vector<DenseLayer> layers;
        std::cout<<"Loading weights.bin...\n";
        load_weights("weights.bin",layers,perm_arena);
        int batch_size=1;
        float* input=compute_arena.allocate<float>(batch_size*layers[0].in_features);
        float* output=compute_arena.allocate<float>(batch_size*layers.back().out_features);
        for(int i=0;i<layers[0].in_features;++i){input[i]=1.0f;}
        std::cout<<"Running AVX2 Inference...\n";
        auto start=std::chrono::high_resolution_clock::now();
        run_inference(layers,input,output,compute_arena,batch_size);
        auto end=std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli>duration=end-start;
        std::cout<<"\n=== HARDWARE METRICS ===\n";
        std::cout<<"Total Inference Latency : "<<duration.count()<<" ms\n";
        std::cout<<"Static Weight Memory    : "<<perm_arena.get_used_bytes()/1024.0<<" KB\n";
        std::cout<<"Ping-Pong Working Space : "<<compute_arena.get_used_bytes()/1024.0<<" KB\n";
        std::cout<<"OS Heap Allocations     : 0 (During execution)\n";
        std::cout<<"Network Output [0]      : "<<output[0]<<"\n";
    }catch(const std::exception& e){
        std::cerr<<"Engine Error: "<<e.what()<<"\n";
    }
    return 0;
}