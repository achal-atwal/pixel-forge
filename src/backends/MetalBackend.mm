#ifdef USE_METAL
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backends/MetalBackend.hpp"
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <string>

// Embedded Metal shader source
static const char* kMetalSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Dims { uint width; uint height; };

kernel void grayscale(
    const device uchar4* in  [[buffer(0)]],
    device       uchar4* out [[buffer(1)]],
    constant     Dims&   dim [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint idx = gid.y * dim.width + gid.x;
    uchar4 p = in[idx];
    uchar luma = (uchar)(0.299f*p.r + 0.587f*p.g + 0.114f*p.b);
    out[idx] = uchar4(luma, luma, luma, p.a);
}

kernel void gaussian_blur(
    const device uchar4* in  [[buffer(0)]],
    device       uchar4* out [[buffer(1)]],
    constant     Dims&   dim [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    const float k[7] = {0.0702f, 0.1311f, 0.1907f, 0.2161f, 0.1907f, 0.1311f, 0.0702f};
    int w=(int)dim.width, h=(int)dim.height;
    float r=0,g=0,b=0;
    for (int dy=-3;dy<=3;dy++) for (int dx=-3;dx<=3;dx++) {
        int nx=clamp((int)gid.x+dx,0,w-1);
        int ny=clamp((int)gid.y+dy,0,h-1);
        uchar4 p=in[ny*w+nx];
        float kw=k[dx+3]*k[dy+3];
        r+=kw*p.r; g+=kw*p.g; b+=kw*p.b;
    }
    uint idx=gid.y*dim.width+gid.x;
    out[idx]=uchar4((uchar)r,(uchar)g,(uchar)b,in[idx].a);
}

kernel void sobel_edge(
    const device uchar4* in  [[buffer(0)]],
    device       uchar4* out [[buffer(1)]],
    constant     Dims&   dim [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    int w=(int)dim.width, h=(int)dim.height;
    int x=(int)gid.x, y=(int)gid.y;
    auto luma = [&](int px, int py) -> float {
        px=clamp(px,0,w-1); py=clamp(py,0,h-1);
        uchar4 p=in[py*w+px];
        return 0.299f*p.r+0.587f*p.g+0.114f*p.b;
    };
    float gx = -luma(x-1,y-1)+luma(x+1,y-1)-2*luma(x-1,y)+2*luma(x+1,y)
               -luma(x-1,y+1)+luma(x+1,y+1);
    float gy = -luma(x-1,y-1)-2*luma(x,y-1)-luma(x+1,y-1)
               +luma(x-1,y+1)+2*luma(x,y+1)+luma(x+1,y+1);
    uchar mag=(uchar)clamp(sqrt(gx*gx+gy*gy),0.f,255.f);
    uint idx=gid.y*dim.width+gid.x;
    out[idx]=uchar4(mag,mag,mag,in[idx].a);
}

kernel void bilateral_filter(
    const device uchar4* in  [[buffer(0)]],
    device       uchar4* out [[buffer(1)]],
    constant     Dims&   dim [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    int w=(int)dim.width, h=(int)dim.height;
    int x=(int)gid.x, y=(int)gid.y;
    const float inv2ss2=1.f/(2.f*25.f), inv2sr2=1.f/(2.f*1600.f);
    uchar4 cp=in[y*w+x];
    float sr=0,sg=0,sb=0,wsum=0;
    for (int dy=-7;dy<=7;dy++) for (int dx=-7;dx<=7;dx++) {
        int nx=clamp(x+dx,0,w-1), ny=clamp(y+dy,0,h-1);
        uchar4 np=in[ny*w+nx];
        float ds=(float)(dx*dx+dy*dy);
        float dr0=(float)cp.r-(float)np.r, dr1=(float)cp.g-(float)np.g,
              dr2=(float)cp.b-(float)np.b;
        float dr=dr0*dr0+dr1*dr1+dr2*dr2;
        float wv=exp(-ds*inv2ss2-dr*inv2sr2);
        sr+=wv*np.r; sg+=wv*np.g; sb+=wv*np.b; wsum+=wv;
    }
    uint idx=gid.y*dim.width+gid.x;
    out[idx]=uchar4((uchar)(sr/wsum),(uchar)(sg/wsum),(uchar)(sb/wsum),cp.a);
}

kernel void histogram_eq(
    const device uchar4* in    [[buffer(0)]],
    device       uchar4* out   [[buffer(1)]],
    constant     Dims&   dim   [[buffer(2)]],
    device const uchar*  lut_r [[buffer(3)]],
    device const uchar*  lut_g [[buffer(4)]],
    device const uchar*  lut_b [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint idx=gid.y*dim.width+gid.x;
    uchar4 p=in[idx];
    out[idx]=uchar4(lut_r[p.r],lut_g[p.g],lut_b[p.b],p.a);
}

kernel void kuwahara(
    const device uchar4* in  [[buffer(0)]],
    device       uchar4* out [[buffer(1)]],
    constant     Dims&   dim [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    int w=(int)dim.width, h=(int)dim.height;
    int x=(int)gid.x, y=(int)gid.y;

    const int qx0[4] = {-5, 0, -5, 0};
    const int qx1[4] = { 0, 5,  0, 5};
    const int qy0[4] = {-5,-5,  0, 0};
    const int qy1[4] = { 0, 0,  5, 5};

    float best_var = 1e30f;
    float best_r = 0, best_g = 0, best_b = 0;

    for (int q = 0; q < 4; q++) {
        float sr=0, sg=0, sb=0, sl=0, sl2=0;
        int count = 0;
        for (int dy = qy0[q]; dy <= qy1[q]; dy++) {
            int ny = clamp(y+dy, 0, h-1);
            for (int dx = qx0[q]; dx <= qx1[q]; dx++) {
                int nx = clamp(x+dx, 0, w-1);
                uchar4 p = in[ny*w+nx];
                float luma = 0.299f*p.r + 0.587f*p.g + 0.114f*p.b;
                sr += p.r; sg += p.g; sb += p.b;
                sl += luma; sl2 += luma*luma;
                count++;
            }
        }
        float inv = 1.f / (float)count;
        float mean_l = sl * inv;
        float var = sl2 * inv - mean_l * mean_l;
        if (var < best_var) {
            best_var = var;
            best_r = sr * inv;
            best_g = sg * inv;
            best_b = sb * inv;
        }
    }

    uint idx = gid.y * dim.width + gid.x;
    out[idx] = uchar4((uchar)clamp((int)best_r, 0, 255),
                      (uchar)clamp((int)best_g, 0, 255),
                      (uchar)clamp((int)best_b, 0, 255),
                      in[idx].a);
}
)METAL";

// pimpl
struct MetalBackendImpl {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    bool ready = false;

    void init() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) return;
        queue  = [device newCommandQueue];

        NSError* err = nil;
        id<MTLLibrary> lib = [device
            newLibraryWithSource:[NSString stringWithUTF8String:kMetalSrc]
            options:nil error:&err];
        if (!lib) { NSLog(@"Metal compile error: %@", err); return; }

        for (NSString* fn in @[@"grayscale",@"gaussian_blur",@"sobel_edge",
                                @"bilateral_filter",@"histogram_eq",@"kuwahara"]) {
            id<MTLFunction> func = [lib newFunctionWithName:fn];
            NSError* e2 = nil;
            id<MTLComputePipelineState> ps =
                [device newComputePipelineStateWithFunction:func error:&e2];
            if (ps) pipelines[[fn UTF8String]] = ps;
            else NSLog(@"Metal pipeline compile failed for %@: %@", fn, e2);
        }
        ready = (pipelines.size() == 6);
    }
};

// MetalBackend
MetalBackend::MetalBackend() : impl_(std::make_unique<MetalBackendImpl>()) {
    impl_->init();
}
MetalBackend::~MetalBackend() = default;
std::string MetalBackend::name()      const { return "metal"; }
bool        MetalBackend::available() const { return impl_->ready; }

BenchmarkResult MetalBackend::run(const IFilter& filter, const Image& input) const {
    if (!impl_->ready)
        throw std::runtime_error("Metal not available");

    auto it = impl_->pipelines.find(filter.name());
    if (it == impl_->pipelines.end())
        throw std::runtime_error("No Metal kernel for filter: " + filter.name());

    int w = input.width(), h = input.height();
    int grid_x = (w + 15) / 16, grid_y = (h + 15) / 16;
    std::cout << "[metal] grid=" << grid_x << "x" << grid_y
              << " (" << grid_x * grid_y * 256 << " GPU threads)\n";
    size_t bytes = input.size();

    id<MTLBuffer> buf_in  = [impl_->device newBufferWithBytes:input.data()
                              length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out = [impl_->device newBufferWithLength:bytes
                              options:MTLResourceStorageModeShared];

    struct Dims { uint32_t width, height; } dims{(uint32_t)w,(uint32_t)h};
    id<MTLBuffer> buf_dims = [impl_->device newBufferWithBytes:&dims
                               length:sizeof(dims) options:MTLResourceStorageModeShared];

    // Histogram equalization needs LUTs precomputed on CPU first
    id<MTLBuffer> buf_lut_r=nil, buf_lut_g=nil, buf_lut_b=nil;
    if (filter.name() == "histogram_eq") {
        auto build_lut = [&](int ch_off) -> std::vector<uint8_t> {
            int hist[256]={};
            for (int i=0;i<w*h;i++) hist[input.data()[i*4+ch_off]]++;
            int cdf[256]={}; cdf[0]=hist[0];
            for (int v=1;v<256;v++) cdf[v]=cdf[v-1]+hist[v];
            int cdf_min=0;
            for (int v=0;v<256;v++) if(cdf[v]>0){cdf_min=cdf[v];break;}
            int denom=w*h-cdf_min;
            std::vector<uint8_t> lut(256);
            for (int v=0;v<256;v++) {
                lut[v]=(denom==0)?v:(uint8_t)std::clamp(
                    (int)std::round((float)(cdf[v]-cdf_min)/(float)denom*255.f),0,255);
            }
            return lut;
        };
        auto lr=build_lut(0), lg=build_lut(1), lb=build_lut(2);
        buf_lut_r=[impl_->device newBufferWithBytes:lr.data() length:256
                   options:MTLResourceStorageModeShared];
        buf_lut_g=[impl_->device newBufferWithBytes:lg.data() length:256
                   options:MTLResourceStorageModeShared];
        buf_lut_b=[impl_->device newBufferWithBytes:lb.data() length:256
                   options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer>       cmd  = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:it->second];
    [enc setBuffer:buf_in  offset:0 atIndex:0];
    [enc setBuffer:buf_out offset:0 atIndex:1];
    [enc setBuffer:buf_dims offset:0 atIndex:2];
    if (filter.name() == "histogram_eq") {
        [enc setBuffer:buf_lut_r offset:0 atIndex:3];
        [enc setBuffer:buf_lut_g offset:0 atIndex:4];
        [enc setBuffer:buf_lut_b offset:0 atIndex:5];
    }

    MTLSize tg  = {16, 16, 1};
    MTLSize grid = {(NSUInteger)((w+15)/16*16), (NSUInteger)((h+15)/16*16), 1};
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    // GPU timing
    __block CFAbsoluteTime t0, t1;
    [cmd addScheduledHandler:^(id<MTLCommandBuffer>){ t0=CFAbsoluteTimeGetCurrent(); }];
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>){ t1=CFAbsoluteTimeGetCurrent(); }];
    [cmd commit];
    [cmd waitUntilCompleted];

    BenchmarkResult result;
    result.backend_name = name();
    result.filter_name  = filter.name();
    result.elapsed_ms   = (float)((t1-t0)*1000.0);
    result.output       = Image(w, h, 4);
    std::memcpy(result.output.data(), [buf_out contents], bytes);
    return result;
}
#endif // USE_METAL
