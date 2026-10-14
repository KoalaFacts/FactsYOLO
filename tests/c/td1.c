/* SPDX-License-Identifier: Apache-2.0 */
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
    if(argc!=4)return 2;
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
    assert(fy_session_release(&session,NULL)==FY_OK && session==NULL);memset(pixels,0,sizeof(pixels));
    fy_result_info info={HEADER(fy_result_info),0,0,0,0,0,0,0,0,0,{0}};
    assert(fy_result_describe(result,&info,NULL)==FY_OK);assert(info.count==expected_count && info.backend_id==FY_BACKEND_TD1_CPU);
    assert(info.source_width==64 && info.source_height==64 && memcmp(info.model_id,artifact+24,32)==0);
    assert(fy_result_copy(result,&actual,1,&count,NULL)==FY_OK && count==expected_count);if(count)compare(&actual,&reference);
    assert(fy_result_release(&result,NULL)==FY_OK && result==NULL);
    puts("PASS C TD1 trained artifact, reference predictions, formats, failures and ownership");return 0;
}
