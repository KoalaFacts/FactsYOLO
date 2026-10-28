/* SPDX-License-Identifier: Apache-2.0 */
/* Experimental Linux x64 timing consumer derived from tests/c/td1.c.
 * Retains every negative/ownership/reference assertion in the original.
 * Never outputs model weights or held-out pixel bytes.
 */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <sys/resource.h>
#include <time.h>
#include <facts_yolo.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HEADER(T) (uint32_t)sizeof(T), FY_STRUCT_VERSION, 0

static size_t read_file(const char *path, unsigned char *bytes, size_t cap) {
    FILE *f=fopen(path,"rb"); assert(f!=NULL);
    size_t n=fread(bytes,1,cap,f); assert(!ferror(f)); assert(fgetc(f)==EOF); assert(fclose(f)==0); return n;
}
static int close_float(float a,float b) {return isfinite(a)&&isfinite(b)&&fabsf(a-b)<=0.0001f+0.0001f*fabsf(b);}
static void compare(const fy_detection *a,const fy_detection *b) {
    assert(a->class_id==b->class_id); assert(close_float(a->score,b->score));
    assert(close_float(a->x_min,b->x_min));assert(close_float(a->y_min,b->y_min));
    assert(close_float(a->x_max,b->x_max));assert(close_float(a->y_max,b->y_max));
}
int main(int argc,char **argv) {
    if(argc!=5)return 2;
    fy_abi_info stamp={HEADER(fy_abi_info),0,0,{0}};
    assert(strlen(argv[4])==40);
    assert(fy_abi_query(&stamp,NULL)==FY_OK && stamp.abi_major==FY_ABI_MAJOR);
    assert(memcmp(stamp.build_id,argv[4],40)==0 && stamp.build_id[40]==0);
    unsigned char artifact[65536],pixels[12288];
    size_t length=read_file(argv[1],artifact,sizeof(artifact));assert(length==51284);
    assert(read_file(argv[2],pixels,sizeof(pixels))==sizeof(pixels));
    FILE *expected=fopen(argv[3],"r");assert(expected!=NULL);
    unsigned int expected_count=99;fy_detection reference={0};
    assert(fscanf(expected,"%u",&expected_count)==1 && expected_count<=1);
    if(expected_count)assert(fscanf(expected,"%u %f %f %f %f %f",&reference.class_id,&reference.score,&reference.x_min,&reference.y_min,&reference.x_max,&reference.y_max)==6);
    assert(fclose(expected)==0);
    size_t count=0;fy_capability capability;
    assert(fy_capabilities_copy(&capability,1,&count,NULL)==FY_OK && count==1);
    assert(capability.backend_id==FY_BACKEND_TD1_CPU && capability.flags==(FY_CAP_COMPILED|FY_CAP_AVAILABLE));
    fy_model_options mo={HEADER(fy_model_options),65536,65536};fy_model *model=NULL;
    artifact[length-1]^=1;assert(fy_model_open(1,artifact,length,&mo,&model,NULL)==FY_UNSUPPORTED_MODEL && model==NULL);artifact[length-1]^=1;
    assert(fy_model_open(1,artifact,length,&mo,&model,NULL)==FY_OK && model!=NULL);
    fy_session_options so={HEADER(fy_session_options),FY_BACKEND_TD1_CPU,FY_DEVICE_CPU,0,FY_PRECISION_F32,FY_FALLBACK_NONE,1,65536,4096,16};
    fy_session *session=NULL;assert(fy_session_create(1,model,&so,&session,NULL)==FY_OK);
    assert(fy_model_release(&model,NULL)==FY_OK && model==NULL);
    fy_detect_options options={HEADER(fy_detect_options),0.5f,16};
    fy_image image={HEADER(fy_image),pixels,sizeof(pixels),63,64,192,FY_RGB8};fy_result *result=NULL;
    assert(fy_detect(session,&image,&options,&result,NULL)==FY_INVALID_ARGUMENT && result==NULL);
    image.width=64;assert(fy_detect(session,&image,&options,&result,NULL)==FY_OK);
    fy_result *saved=result;
    assert(fy_detect(session,&image,&options,&result,NULL)==FY_INVALID_ARGUMENT && result==saved);
    fy_detection actual;count=99;assert(fy_result_copy(result,&actual,1,&count,NULL)==FY_OK && count==expected_count);
    if(count)compare(&actual,&reference);
    for(unsigned int format=FY_BGR8;format<=FY_RGBA8;++format) {
        size_t channels=format==FY_RGBA8?4:3,stride=64*channels+7;
        unsigned char *other=malloc(stride*64);assert(other!=NULL);memset(other,199,stride*64);
        for(size_t y=0;y<64;++y)for(size_t x=0;x<64;++x) {
            const unsigned char *p=pixels+(y*64+x)*3;unsigned char *q=other+y*stride+x*channels;
            q[0]=p[format==FY_BGR8?2:0];q[1]=p[1];q[2]=p[format==FY_BGR8?0:2];
        }
        fy_image alternate={HEADER(fy_image),other,stride*64,64,64,stride,format};fy_result *next=NULL;
        assert(fy_detect(session,&alternate,&options,&next,NULL)==FY_OK);free(other);
        assert(fy_result_copy(next,&actual,1,&count,NULL)==FY_OK && count==expected_count);if(count)compare(&actual,&reference);
        assert(fy_result_release(&next,NULL)==FY_OK && next==NULL);
    }
    if(!expected_count || reference.score<1.0f) {
        options.score_threshold=1.0f;fy_result *filtered=NULL;
        assert(fy_detect(session,&image,&options,&filtered,NULL)==FY_OK);
        assert(fy_result_copy(filtered,NULL,0,&count,NULL)==FY_OK && count==0);
        assert(fy_result_release(&filtered,NULL)==FY_OK);
    }
    
    /* Model loading, prior conformance checks and warm-up are excluded. The
     * measured scope contains only C detect/result copy/release. Each result
     * still matches the independently trained portable reference. */
    for(int iteration=0;iteration<64;++iteration) {
        fy_result *warm=NULL;fy_detection d={0};size_t n=0;
        assert(fy_detect(session,&image,&(fy_detect_options){HEADER(fy_detect_options),0.5f,16},&warm,NULL)==FY_OK && warm);
        assert(fy_result_copy(warm,&d,1,&n,NULL)==FY_OK && n==expected_count);
        if(n)compare(&d,&reference);
        assert(fy_result_release(&warm,NULL)==FY_OK && warm==NULL);
    }
    struct timespec wall_start,wall_end,cpu_start,cpu_end;
    assert(clock_gettime(CLOCK_MONOTONIC_RAW,&wall_start)==0);
    assert(clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&cpu_start)==0);
    for(int iteration=0;iteration<512;++iteration) {
        fy_result *timed=NULL;fy_detection d={0};size_t n=0;
        assert(fy_detect(session,&image,&(fy_detect_options){HEADER(fy_detect_options),0.5f,16},&timed,NULL)==FY_OK && timed);
        assert(fy_result_copy(timed,&d,1,&n,NULL)==FY_OK && n==expected_count);
        if(n)compare(&d,&reference);
        assert(fy_result_release(&timed,NULL)==FY_OK && timed==NULL);
    }
    assert(clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&cpu_end)==0);
    assert(clock_gettime(CLOCK_MONOTONIC_RAW,&wall_end)==0);
    uint64_t wall_ns=(uint64_t)(wall_end.tv_sec-wall_start.tv_sec)*UINT64_C(1000000000)+(uint64_t)(wall_end.tv_nsec-wall_start.tv_nsec);
    uint64_t cpu_ns=(uint64_t)(cpu_end.tv_sec-cpu_start.tv_sec)*UINT64_C(1000000000)+(uint64_t)(cpu_end.tv_nsec-cpu_start.tv_nsec);
    uint32_t prediction_bits[6]={0};
    if(expected_count) {
        prediction_bits[0]=actual.class_id;
        memcpy(&prediction_bits[1],&actual.score,sizeof(float));
        memcpy(&prediction_bits[2],&actual.x_min,sizeof(float));
        memcpy(&prediction_bits[3],&actual.y_min,sizeof(float));
        memcpy(&prediction_bits[4],&actual.x_max,sizeof(float));
        memcpy(&prediction_bits[5],&actual.y_max,sizeof(float));
    }
    struct rusage usage;assert(getrusage(RUSAGE_SELF,&usage)==0);
    printf("{\"calls\":512,\"wall_ns\":%"PRIu64",\"cpu_ns\":%"PRIu64",\"count\":%u,\"prediction_bits\":[%u,%u,%u,%u,%u,%u],\"process_maxrss_kib\":%ld}\n",
        wall_ns,cpu_ns,expected_count,
        prediction_bits[0],prediction_bits[1],prediction_bits[2],
        prediction_bits[3],prediction_bits[4],prediction_bits[5],usage.ru_maxrss);
    fflush(stdout);
    assert(fy_session_release(&session,NULL)==FY_OK && session==NULL);memset(pixels,0,sizeof(pixels));
    fy_result_info info={HEADER(fy_result_info),0,0,0,0,0,0,0,0,0,{0}};
    assert(fy_result_describe(result,&info,NULL)==FY_OK);assert(info.count==expected_count && info.backend_id==FY_BACKEND_TD1_CPU);
    assert(info.source_width==64 && info.source_height==64 && memcmp(info.model_id,artifact+24,32)==0);
    assert(fy_result_copy(result,&actual,1,&count,NULL)==FY_OK && count==expected_count);if(count)compare(&actual,&reference);
    assert(fy_result_release(&result,NULL)==FY_OK && result==NULL);
    return 0;
}
