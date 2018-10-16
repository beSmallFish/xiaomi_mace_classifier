#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv/cv.h>
#include "thread_event.h"

#include <src/main/cpp/image_classify.h>


#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "error", __VA_ARGS__))
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "debug", __VA_ARGS__))

using namespace cv;
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

static pthread_t __pthread_ptr;
static struct thread_event_t __thread_event;
static timeval image_handle_end, image_handle_start,
        image2mat_end, image2mat_start,
        rgb2yCbcr_end, rgb2yCbcr_start,
        rgb_split_end, rgb_split_start,
        y_output_process_end, y_output_process_start,
        mace_end, mace_start,
        y_input_process_end, y_input_process_start,
        cbs_end, cbs_start,
        crs_end, crs_start,
        yCbcr_merge_end, yCbcr_merge_start,
        yCbcr2rgb_end, yCbcr2rgb_start,
        mat2image_end, mat2image_start;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    //__java_vm_ptr = vm;
    JNIEnv* env = NULL;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_4) == JNI_EDETACHED) {
        return -1;
    }
   // if (blackwidow_init(env) != 0) {
   //     return -2;
   // }

    if (thread_event_init(__thread_event) < 0) {
        return -2;
    }

    return JNI_VERSION_1_4;
}

typedef struct _CbCr_process {
    timeval* time_start;
    timeval* time_end;
    Mat input_mat;
    Mat output_mat;
} CbCr_process;

typedef struct _Y_process {
    timeval* mace_start;
    timeval* mace_end;
    timeval* y_input_process_start;
    timeval* y_input_process_end;
    timeval* y_output_process_start;
    timeval* y_output_process_end;
    Mat input_mat;
    Mat output_mat;
} Y_process;


typedef struct _MACE_process {
    timeval* time_start;
    timeval* time_end;
    Mat input_mat;
    Mat output_mat;
} MACE_process;

static Mat *mat(jlong nativeObj)
{
    return reinterpret_cast<Mat *>(nativeObj);
}

jlong getNativeObj(Mat mat){
    return reinterpret_cast<jlong>(new Mat(mat));
}

void BitmapToMat2(JNIEnv *env, jobject &bitmap, Mat &mat, jboolean needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &dst = mat;

    try {
        LOGD("nBitmapToMat");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        dst.create(info.height, info.width, CV_8UC4);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGD("nBitmapToMat: RGBA_8888 -> CV_8UC4");
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (needUnPremultiplyAlpha) cvtColor(tmp, dst, COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
            LOGD("nBitmapToMat: RGBA_8888 -> CV_8UC4 ===> 1");
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            LOGD("nBitmapToMat: RGB_565 -> CV_8UC4");
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

void BitmapToMat(JNIEnv *env, jobject &bitmap, Mat &mat) {
    BitmapToMat2(env, bitmap, mat, false);
}

void MatToBitmap2
        (JNIEnv *env, Mat &mat, jobject &bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &src = mat;

    try {
        LOGD("nMatToBitmap");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        LOGD("nMatToBitmap: info.height:%d, src.rows:%d, info.width:%d, src.cols:%d", info.height,
             src.rows, info.width, src.cols);
        CV_Assert(src.dims == 2 && info.height == (uint32_t) src.rows &&
                  info.width == (uint32_t) src.cols);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                LOGD("nMatToBitmap: CV_8UC1 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                LOGD("nMatToBitmap: CV_8UC3 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if (src.type() == CV_8UC4) {
                LOGD("nMatToBitmap: CV_8UC4 -> RGBA_8888");
                if (needPremultiplyAlpha)
                    cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else
                    src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                LOGD("nMatToBitmap: CV_8UC1 -> RGB_565");
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                LOGD("nMatToBitmap: CV_8UC3 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                LOGD("nMatToBitmap: CV_8UC4 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nMatToBitmap catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nMatToBitmap catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}

void MatToBitmap(JNIEnv *env, Mat &mat, jobject &bitmap) {
    MatToBitmap2(env, mat, bitmap, false);
}

int timeDiff(timeval end, timeval start) {
    // gettimeofday( &start, NULL );
    // gettimeofday( &end, NULL );
    int timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // printf("time: %d us\n", timeuse);
    return timeuse / 1000;
}

void printDiff(string tag){
    LOGD("%s  totalTime:%d, image2mat:%d, rgb2yCbCr:%d, rgb_split:%d, y_mace_input_process:%d, mace:%d, "
                 "y_mace_output_process:%d, Cb_process:%d, Cr_process:%d, yCbCr_merge:%d, yCbCr2rgb:%d, mat2image:%d",
         tag.c_str(),
         timeDiff(image_handle_end, image_handle_start),
         timeDiff(image2mat_end, image2mat_start),
         timeDiff(rgb2yCbcr_end, rgb2yCbcr_start),
         timeDiff(rgb_split_end, rgb_split_start),
         timeDiff(y_input_process_end, y_input_process_start),
         timeDiff(mace_end, mace_start),
         timeDiff(y_output_process_end, y_output_process_start),
         timeDiff(cbs_end, cbs_start),
         timeDiff(crs_end, crs_start),
         timeDiff(yCbcr_merge_end, yCbcr_merge_start),
         timeDiff(yCbcr2rgb_end, yCbcr2rgb_start),
         timeDiff(mat2image_end, mat2image_start)
    );
}


JNIEXPORT void JNICALL
Java_com_spencerfricke_opencv_1ndk_MainActivity_onCreateJNI(
        JNIEnv *env, jobject clazz, jobject activity, jobject j_asset_manager) {

}

// 定义调用java中的方法的函数
void callJavaMethod( JNIEnv *env, jobject thiz, jfloatArray input, jfloatArray& output, jint width, jint height){
    // 先找到要调用的类
    jclass clazz = env -> FindClass("gdut/bsx/tensorflowtraining/OpenCVUtil");
    if (clazz == NULL){
        printf("find class OpenCVUtil error !");
        return;
    }
    // 获取java方法id
    // 参数二是调用的方法名,  参数三是方法的签名
    jmethodID id = env -> GetStaticMethodID(clazz, "callMaceProcess", "([FII)[F");
    if (id == NULL){
        printf("find method callMaceProcess error !");
        return;
    }
    //jstring msg = env->NewStringUTF("msg send by callJavaMethod in test.cpp .");
    // 开始调用java中的静态方法
    output  = (jfloatArray)(env -> CallStaticObjectMethod(clazz, id, input, width, height));
}

// 定义调用java中的方法的函数 已弃用
void callMaceProcess2( JNIEnv *env, jobject thiz, Mat input_mat, Mat& output_mat){
    // 先找到要调用的类
    jclass clazz = env -> FindClass("gdut/bsx/tensorflowtraining/OpenCVUtil");
    if (clazz == NULL){
        printf("find class OpenCVUtil error !");
        return;
    }
    // 获取java方法id
    // 参数二是调用的方法名,  参数三是方法的签名
    jmethodID id = env -> GetStaticMethodID(clazz, "callMaceProcess2", "(J)J");
    if (id == NULL){
        printf("find method callMaceProcess error !");
        return;
    }
    //jstring msg = env->NewStringUTF("msg send by callJavaMethod in test.cpp .");
    // 开始调用java中的静态方法
    jlong output_nativeObj  = env -> CallStaticLongMethod(clazz, id, getNativeObj(input_mat));
    output_mat = *mat(output_nativeObj);
}

// 定义调用java中的方法的函数 已弃用
void callMaceProcess3( JNIEnv *env, jobject thiz, jfloatArray input_arr, Mat& output_mat, int width, int height){
    // 先找到要调用的类
    jclass clazz = env -> FindClass("gdut/bsx/tensorflowtraining/OpenCVUtil");
    if (clazz == NULL){
        printf("find class OpenCVUtil error !");
        return;
    }
    // 获取java方法id
    // 参数二是调用的方法名,  参数三是方法的签名
    jmethodID id = env -> GetStaticMethodID(clazz, "callMaceProcess3", "([FII)J");
    if (id == NULL){
        printf("find method callMaceProcess error !");
        return;
    }
    //jstring msg = env->NewStringUTF("msg send by callJavaMethod in test.cpp .");
    // 开始调用java中的静态方法
    jlong output_nativeObj  = env -> CallStaticLongMethod(clazz, id, input_arr, width, height);
    output_mat = *mat(output_nativeObj);
}

// 定义调用java中的方法的函数 已弃用
void callJavaCubicMethod(JNIEnv *env, jobject thiz, Mat input, Mat &output, jint width, jint height){
    // 先找到要调用的类
    jclass clazz = env -> FindClass("gdut/bsx/tensorflowtraining/OpenCVUtil");
    if (clazz == NULL){
        printf("find class OpenCVUtil error !");
        return;
    }
    // 获取java方法id
    // 参数二是调用的方法名,  参数三是方法的签名
    jmethodID id = env -> GetStaticMethodID(clazz, "callCubicProcess", "(Lorg/opencv/core/Mat;II)[B");
    if (id == NULL){
        printf("find method callMaceProcess error !");
        return;
    }
    //jstring msg = env->NewStringUTF("msg send by callJavaMethod in test.cpp .");

    jclass Mat_clazz = env -> FindClass("org/opencv/core/Mat");//获得ArrayList类引用
    //获得得该类型的构造函数  函数名为 <init> 返回类型必须为 void 即 V
    jmethodID constrocMID = env->GetMethodID(Mat_clazz,"<init>","(III)V");
    // 开始调用java中的静态方法
    //Mat cb_input_mat = new Mat(x , y, CvType.CV_8U);  cb_input_mat.put(0,0, cb_input);
    LOGD("test !!!! : callJavaCubicMethod ===> 1!! ");
    jobject input_mat_obj = env->NewObject(Mat_clazz,constrocMID, width/3, height/3, CV_8U);  //构造一个对象，调用该类的构造函数，并且传递参数
    jmethodID method_mat_put = env -> GetMethodID(Mat_clazz, "put", "(II[B)I");
//    LOGD("test !!!! : callJavaCubicMethod ===> 2!! input_mat_obj:%p, %p, %p, %d", &input_mat_obj, &method_mat_put, &input, &input);

    jbyteArray input_array = env -> NewByteArray(width * height / 9);
    jbyte * short_ptr = (jbyte *)input.data;
    LOGD("test !!!! : callJavaCubicMethod ===> 3_1!! short_ptr:%p", short_ptr);
    env->SetByteArrayRegion(input_array, 0, env->GetArrayLength(input_array), short_ptr);
    LOGD("test !!!! : callJavaCubicMethod ===> 3_2!! ");
    env->CallIntMethod(input_mat_obj, method_mat_put, 0, 0, input_array);
    LOGD("test !!!! : callJavaCubicMethod ===> 3!! ");
    jbyteArray output_array  = (jbyteArray)(env -> CallStaticObjectMethod(clazz, id, input_mat_obj, width, height));
    jbyte * output_ptr = env->GetByteArrayElements(output_array, 0);

    //jshortArray output_array  = (jshortArray)(env -> CallStaticObjectMethod(clazz, id, input_mat_obj, width, height));
   // jshort * output_ptr = env->GetShortArrayElements(output_array, 0);

    //jintArray output_array  = (jintArray)(env -> CallStaticObjectMethod(clazz, id, input_mat_obj, width, height));
    //jint * output_ptr = env->GetIntArrayElements(output_array, 0);

   // Mat mace_y_output(newRows, newCols, CV_32F, (unsigned char*)aa);
    output.data = (unsigned char *)output_ptr;


   // env->DeleteLocalRef(input_mat_obj);
   // env->DeleteLocalRef(input_array);
   // env->DeleteLocalRef(output_array);
}

void callJavaCubicMethod2(JNIEnv *env, jobject thiz, Mat input, Mat &output, jint width, jint height){
    // 先找到要调用的类
    jclass clazz = env -> FindClass("gdut/bsx/tensorflowtraining/OpenCVUtil");
    if (clazz == NULL){
        printf("find class OpenCVUtil error !");
        return;
    }
    // 获取java方法id
    // 参数二是调用的方法名,  参数三是方法的签名
    jmethodID id = env -> GetStaticMethodID(clazz, "callCubicProcess2", "(JII)J");
    if (id == NULL){
        printf("find method callMaceProcess error !");
        return;
    }

    jlong input_nativeObj = getNativeObj(input);
    jlong output_nativeObj = env -> CallStaticLongMethod(clazz, id, input_nativeObj, width, height);

    output = *mat(output_nativeObj);
    //LOGD("test !!!! : callJavaCubicMethod ===> 4!! %d %d", input_nativeObj, output_nativeObj);
}

JNIEXPORT void JNICALL Java_gdut_bsx_tensorflowtraining_OpenCVUtil_processImage(
        JNIEnv *env, jobject clazz, jobject jsrcBitmap, jobject jdstBitmap) {

    gettimeofday(&image_handle_start, NULL) ;

    gettimeofday(&image2mat_start, NULL ) ;
    Mat mat_src_rgb;
    BitmapToMat(env, jsrcBitmap, mat_src_rgb);//图片转化成mat
    int newRows = mat_src_rgb.rows * 3;
    int newCols = mat_src_rgb.cols * 3;
    gettimeofday(&image2mat_end, NULL ) ;

    gettimeofday(&rgb2yCbcr_start, NULL ) ;
    Mat mat_dst_rgb;
    Mat mat_src_ycbcr;
    cvtColor(mat_src_rgb, mat_src_ycbcr, COLOR_RGB2YCrCb);
    gettimeofday(&rgb2yCbcr_end, NULL ) ;

//    LOGD("nBitmapToMat: COLOR_RGB2YCrCb ===> end!! mat_src_ycbcr:%d", mat_src_ycbcr);
    gettimeofday(&rgb_split_start, NULL ) ;
    Mat ycbcr_arr[3];
    split(mat_src_ycbcr, ycbcr_arr);
    Mat y = ycbcr_arr[0];
    Mat cb = ycbcr_arr[1];
    Mat cr = ycbcr_arr[2];
    gettimeofday(&rgb_split_end, NULL ) ;

    gettimeofday(&y_input_process_start, NULL ) ;
    float alpha = 2/255.0f;
    Mat mace_y_input;
    //cvConvertScale(&y, &mace_y_input, alpha, -1);
    y.convertTo(mace_y_input, CV_32F, alpha, -1);
   // LOGD("nBitmapToMat: cvConvertScale ===> end!! ");
    gettimeofday(&y_input_process_end, NULL ) ;


    gettimeofday(&mace_start, NULL ) ;
    unsigned char *array = new unsigned char[mace_y_input.rows * mace_y_input.cols];
    if (mace_y_input.isContinuous())
        array = mace_y_input.data;
    jfloatArray mace_y_input_arr = env -> NewFloatArray(mace_y_input.rows * mace_y_input.cols);
    jfloatArray mace_y_output_arr = env -> NewFloatArray(newRows * newCols);
    float* float_arr = (float *)array;
    env->SetFloatArrayRegion(mace_y_input_arr, 0, env->GetArrayLength(mace_y_input_arr), float_arr);
    callJavaMethod(env, clazz, mace_y_input_arr, mace_y_output_arr, mat_src_rgb.cols, mat_src_rgb.rows);
    gettimeofday(&mace_end, NULL ) ;

    gettimeofday(&y_output_process_start, NULL);
    jfloat * aa = env->GetFloatArrayElements(mace_y_output_arr, 0);
    Mat mace_y_output(newRows, newCols, CV_32F, (unsigned char*)aa);
   // LOGD("nBitmapToMat: callJavaMethod ===> end!! mace_y_output_arr:%f, %f",  aa[22], aa[1000]);
    Mat y_output(newRows, newCols, CV_8U);
    float beta = 255/2.0f;
    //cvConvertScale(&mace_y_output, &y_output, beta, beta);
    mace_y_output.convertTo(y_output, CV_8U, beta, beta);
    gettimeofday(&y_output_process_end, NULL);


   gettimeofday(&cbs_start, NULL);
    Mat cb_output(newRows, newCols, CV_8U);
    resize(cb, cb_output, cb_output.size(),3, 3, INTER_CUBIC);
    gettimeofday(&cbs_end, NULL);

    gettimeofday(&crs_start, NULL);
    Mat cr_output(newRows, newCols, CV_8U);
    resize(cr, cr_output, cr_output.size(), 3, 3, INTER_CUBIC);
    gettimeofday(&crs_end, NULL);


    gettimeofday(&yCbcr_merge_start, NULL);
    Mat mat_dst_ycbcr;
    Mat mat_dst_ycbcr_channels[3];
   // LOGD("nBitmapToMat: merge ===> end!! %d, %d, %d", y_output.rows*y_output.cols, cb_output.rows*cb_output.cols, cr_output.rows*cr_output.cols);
    mat_dst_ycbcr_channels[0] = y_output;
    mat_dst_ycbcr_channels[1] = cb_output;
    mat_dst_ycbcr_channels[2] = cr_output;
    merge(mat_dst_ycbcr_channels, 3, mat_dst_ycbcr);
    gettimeofday(&yCbcr_merge_end, NULL);

    gettimeofday(&yCbcr2rgb_start, NULL);
    cvtColor(mat_dst_ycbcr, mat_dst_rgb, COLOR_YCrCb2RGB);
    gettimeofday(&yCbcr2rgb_end, NULL);

    gettimeofday(&mat2image_start, NULL);
    MatToBitmap(env, mat_dst_rgb, jdstBitmap);//mat转成化图片
    gettimeofday(&mat2image_end, NULL);

    gettimeofday(&image_handle_end, NULL);

    printDiff("C image process end !!");

    // return mat_dst_rgb;
}


JNIEXPORT void JNICALL Java_gdut_bsx_tensorflowtraining_OpenCVUtil_processImageMix(
        JNIEnv *env, jobject clazz, jobject jsrcBitmap, jobject jdstBitmap) {

    gettimeofday(&image_handle_start, NULL) ;

    gettimeofday(&image2mat_start, NULL ) ;
    Mat mat_src_rgb;
    BitmapToMat(env, jsrcBitmap, mat_src_rgb);//图片转化成mat
    int newRows = mat_src_rgb.rows * 3;
    int newCols = mat_src_rgb.cols * 3;
    gettimeofday(&image2mat_end, NULL ) ;

    gettimeofday(&rgb2yCbcr_start, NULL ) ;
    Mat mat_dst_rgb;
    Mat mat_src_ycbcr;
    cvtColor(mat_src_rgb, mat_src_ycbcr, COLOR_RGB2YCrCb);
    gettimeofday(&rgb2yCbcr_end, NULL ) ;

//    LOGD("nBitmapToMat: COLOR_RGB2YCrCb ===> end!! mat_src_ycbcr:%d", mat_src_ycbcr);
    gettimeofday(&rgb_split_start, NULL ) ;
    Mat ycbcr_arr[3];
    split(mat_src_ycbcr, ycbcr_arr);
    Mat y = ycbcr_arr[0];
    Mat cb = ycbcr_arr[1];
    Mat cr = ycbcr_arr[2];
    gettimeofday(&rgb_split_end, NULL ) ;

    gettimeofday(&y_input_process_start, NULL ) ;
    float alpha = 2/255.0f;
    Mat mace_y_input;
    //cvConvertScale(&y, &mace_y_input, alpha, -1);
    y.convertTo(mace_y_input, CV_32F, alpha, -1);
   // LOGD("nBitmapToMat: cvConvertScale ===> end!! ");
    gettimeofday(&y_input_process_end, NULL ) ;


   /* gettimeofday(&mace_start, NULL ) ;
    unsigned char *array = new unsigned char[mace_y_input.rows * mace_y_input.cols];
    if (mace_y_input.isContinuous())
        array = mace_y_input.data;
    jfloatArray mace_y_input_arr = env -> NewFloatArray(mace_y_input.rows * mace_y_input.cols);
    jfloatArray mace_y_output_arr = env -> NewFloatArray(newRows * newCols);
    float* float_arr = (float *)array;
    env->SetFloatArrayRegion(mace_y_input_arr, 0, env->GetArrayLength(mace_y_input_arr), float_arr);
    callJavaMethod(env, clazz, mace_y_input_arr, mace_y_output_arr, mat_src_rgb.cols, mat_src_rgb.rows);
    gettimeofday(&mace_end, NULL ) ;*/


    gettimeofday(&mace_start, NULL ) ;
    //Mat mace_y_output;
   // callMaceProcess2(env, clazz, mace_y_input, mace_y_output);
    Mat mace_y_output = maceByMat(mace_y_input, 1);
    gettimeofday(&mace_end, NULL ) ;


    gettimeofday(&y_output_process_start, NULL);
    //jfloat * aa = env->GetFloatArrayElements(mace_y_output_arr, 0);
   // Mat mace_y_output(newRows, newCols, CV_32F, (unsigned char*)aa);
    Mat y_output(newRows, newCols, CV_8U);
    float beta = 255/2.0f;
    mace_y_output.convertTo(y_output, CV_8U, beta, beta);
    gettimeofday(&y_output_process_end, NULL);


    /* gettimeofday(&cbs_start, NULL);
     Mat cb_output(newRows, newCols, CV_8U);
     resize(cb, cb_output, cb_output.size(),3, 3, INTER_CUBIC);
     gettimeofday(&cbs_end, NULL);*/

    gettimeofday(&cbs_start, NULL);
    Mat cb_output(newRows, newCols, CV_8U);
    callJavaCubicMethod2(env, clazz, cb, cb_output, newCols, newRows);
    //resize(cb, cb_output, cb_output.size(),3, 3, INTER_CUBIC);
    gettimeofday(&cbs_end, NULL);


    gettimeofday(&crs_start, NULL);
    Mat cr_output(newRows, newCols, CV_8U);
   // resize(cr, cr_output, cr_output.size(), 3, 3, INTER_CUBIC);
    callJavaCubicMethod2(env, clazz, cr, cr_output, newCols, newRows);
    gettimeofday(&crs_end, NULL);

    gettimeofday(&yCbcr_merge_start, NULL);
    Mat mat_dst_ycbcr;
    Mat mat_dst_ycbcr_channels[3];
    gettimeofday(&yCbcr_merge_start, NULL);
   // LOGD("nBitmapToMat: merge ===> end!! %d, %d, %d", y_output.rows*y_output.cols, cb_output.rows*cb_output.cols, cr_output.rows*cr_output.cols);
    mat_dst_ycbcr_channels[0] = y_output;
    mat_dst_ycbcr_channels[1] = cb_output;
    mat_dst_ycbcr_channels[2] = cr_output;
    merge(mat_dst_ycbcr_channels, 3, mat_dst_ycbcr);
    gettimeofday(&yCbcr_merge_end, NULL);

    gettimeofday(&yCbcr2rgb_start, NULL);
    cvtColor(mat_dst_ycbcr, mat_dst_rgb, COLOR_YCrCb2RGB);
    gettimeofday(&yCbcr2rgb_end, NULL);

    gettimeofday(&mat2image_start, NULL);
    MatToBitmap(env, mat_dst_rgb, jdstBitmap);//mat转成化图片
    gettimeofday(&mat2image_end, NULL);

    gettimeofday(&image_handle_end, NULL);

    printDiff("mix image process end !!");
    // return mat_dst_rgb;
}


void* CbCrCalculate(void* arg){

    LOGE("test !!!! : CbCrCalculate ===> Running");

    CbCr_process *ptr = (CbCr_process *) arg;
    CbCr_process cb_process = ptr[0];
    gettimeofday(cb_process.time_start, NULL);
    resize(cb_process.input_mat, cb_process.output_mat, cb_process.output_mat.size(),3, 3, INTER_CUBIC);
    gettimeofday(cb_process.time_end, NULL);


    CbCr_process cr_process = ptr[1];
    gettimeofday(cr_process.time_start, NULL);
    resize(cr_process.input_mat, cr_process.output_mat, cr_process.output_mat.size(),3, 3, INTER_CUBIC);
    gettimeofday(cr_process.time_end, NULL);

    thread_event_signal(__thread_event);
    LOGE("thread_event_signal -> %d", __thread_event.value);
    LOGE("test !!!! : CbCrCalculate ===> Running end");
    return NULL;
}

void* YCalculate(void* arg){

    LOGE("test !!!! : YCalculate ===> Running");

    Y_process y_process = *(Y_process *)arg;

    gettimeofday(y_process.y_input_process_start, NULL ) ;
    float alpha = 2/255.0f;
    Mat mace_y_input;
    y_process.input_mat.convertTo(mace_y_input, CV_32F, alpha, -1);
    gettimeofday(y_process.y_input_process_end, NULL ) ;

    gettimeofday(y_process.mace_start, NULL ) ;
    Mat mace_y_output = maceByMat(mace_y_input, 1);
    gettimeofday(y_process.mace_end, NULL ) ;

    gettimeofday(y_process.y_output_process_start, NULL);
    //Mat y_output(mace_y_output.rows, mace_y_output.cols, CV_8U);
    float beta = 255/2.0f;
    mace_y_output.convertTo(y_process.output_mat, CV_8U, beta, beta);
    gettimeofday(y_process.y_output_process_end, NULL);

    thread_event_signal(__thread_event);
    LOGE("thread_event_signal -> %d", __thread_event.value);
    LOGE("test !!!! : YCalculate ===> Running end");
    return NULL;
}

void* maceCalculate(void* arg){

    LOGE("test !!!! : maceCalculate ===> Running");

    MACE_process y_process = *(MACE_process *)arg;
    gettimeofday(y_process.time_start, NULL ) ;
    Mat output_mat = maceByMat(y_process.input_mat, 1);
    //y_process.output_mat = output_mat;
    (*(MACE_process *)arg).output_mat = output_mat;
    gettimeofday(y_process.time_end, NULL ) ;

   // LOGE("merge -> %f %f %f", y_process.input_mat.at<float >(1,1),y_process.input_mat.at<float >(122,122), y_process.input_mat.at<float >(333,333));
  //  LOGE("merge2 -> %f %f %f", output_mat.at<float >(1,1),output_mat.at<float >(122,122), output_mat.at<float >(333,333));
    LOGE("test !!!! : maceCalculate ===> Running end");

    LOGE("thread_event_signal -> %d", __thread_event.value);
    thread_event_signal(__thread_event);
    return NULL;
}

JNIEXPORT void JNICALL Java_gdut_bsx_tensorflowtraining_OpenCVUtil_processImageByCbCrThread(
        JNIEnv *env, jobject clazz, jobject jsrcBitmap, jobject jdstBitmap) {

    gettimeofday(&image_handle_start, NULL) ;

    gettimeofday(&image2mat_start, NULL ) ;
    Mat mat_src_rgb;
    BitmapToMat(env, jsrcBitmap, mat_src_rgb);//图片转化成mat
    int newRows = mat_src_rgb.rows * 3;
    int newCols = mat_src_rgb.cols * 3;
    gettimeofday(&image2mat_end, NULL ) ;

    gettimeofday(&rgb2yCbcr_start, NULL ) ;
    Mat mat_dst_rgb;
    Mat mat_src_ycbcr;
    cvtColor(mat_src_rgb, mat_src_ycbcr, COLOR_RGB2YCrCb);
    gettimeofday(&rgb2yCbcr_end, NULL ) ;

//    LOGD("nBitmapToMat: COLOR_RGB2YCrCb ===> end!! mat_src_ycbcr:%d", mat_src_ycbcr);
    gettimeofday(&rgb_split_start, NULL ) ;
    Mat ycbcr_arr[3];
    split(mat_src_ycbcr, ycbcr_arr);
    Mat y = ycbcr_arr[0];
    Mat cb = ycbcr_arr[1];
    Mat cr = ycbcr_arr[2];
    gettimeofday(&rgb_split_end, NULL ) ;

    //开启线程
    CbCr_process args[2];
    Mat cb_output(newRows, newCols, CV_8U);
    CbCr_process cb_process = {&cbs_start, &cbs_end, cb, cb_output};
    Mat cr_output(newRows, newCols, CV_8U);
    CbCr_process cr_process = {&crs_start, &crs_end, cr, cr_output};
    args[0] = cb_process;
    args[1] = cr_process;
    if(pthread_create(&__pthread_ptr, NULL, CbCrCalculate, (void *)args) < 0){
        LOGD("processImageByCbCrThread: 开启线程失败!! ------------------------> 串行执行");
        CbCrCalculate((void *)args);
     //   return;
    }

    gettimeofday(&y_input_process_start, NULL ) ;
    float alpha = 2/255.0f;
    Mat mace_y_input;
    //cvConvertScale(&y, &mace_y_input, alpha, -1);
    y.convertTo(mace_y_input, CV_32F, alpha, -1);
    gettimeofday(&y_input_process_end, NULL ) ;

    gettimeofday(&mace_start, NULL ) ;
    Mat mace_y_output = maceByMat(mace_y_input, 1);
    gettimeofday(&mace_end, NULL ) ;

    gettimeofday(&y_output_process_start, NULL);
  //  jfloat * aa = env->GetFloatArrayElements(mace_y_output_arr, 0);
   // Mat mace_y_output(newRows, newCols, CV_32F, (unsigned char*)aa);
    Mat y_output(newRows, newCols, CV_8U);
    float beta = 255/2.0f;
    mace_y_output.convertTo(y_output, CV_8U, beta, beta);
    gettimeofday(&y_output_process_end, NULL);

    thread_event_wait(__thread_event);
    LOGE("thread_event_wait -> %d", __thread_event.value);

    //test2
    gettimeofday(&yCbcr_merge_start, NULL);
    Mat mat_dst_ycbcr;
    Mat mat_dst_ycbcr_channels[3];
    gettimeofday(&yCbcr_merge_start, NULL);
    mat_dst_ycbcr_channels[0] = y_output;
    mat_dst_ycbcr_channels[1] = cb_output;
    mat_dst_ycbcr_channels[2] = cr_output;
    merge(mat_dst_ycbcr_channels, 3, mat_dst_ycbcr);
    gettimeofday(&yCbcr_merge_end, NULL);

    gettimeofday(&yCbcr2rgb_start, NULL);
    cvtColor(mat_dst_ycbcr, mat_dst_rgb, COLOR_YCrCb2RGB);
    gettimeofday(&yCbcr2rgb_end, NULL);

    gettimeofday(&mat2image_start, NULL);
    MatToBitmap(env, mat_dst_rgb, jdstBitmap);//mat转成化图片
    gettimeofday(&mat2image_end, NULL);

    gettimeofday(&image_handle_end, NULL);

    printDiff("CbCr_thread image process end !!");
    // return mat_dst_rgb;
}


JNIEXPORT void JNICALL Java_gdut_bsx_tensorflowtraining_OpenCVUtil_processImageByYThread(
        JNIEnv *env, jobject clazz, jobject jsrcBitmap, jobject jdstBitmap) {

    gettimeofday(&image_handle_start, NULL) ;

    gettimeofday(&image2mat_start, NULL ) ;
    Mat mat_src_rgb;
    BitmapToMat(env, jsrcBitmap, mat_src_rgb);//图片转化成mat
    int newRows = mat_src_rgb.rows * 3;
    int newCols = mat_src_rgb.cols * 3;
    gettimeofday(&image2mat_end, NULL ) ;

    gettimeofday(&rgb2yCbcr_start, NULL ) ;
    Mat mat_dst_rgb;
    Mat mat_src_ycbcr;
    cvtColor(mat_src_rgb, mat_src_ycbcr, COLOR_RGB2YCrCb);
    gettimeofday(&rgb2yCbcr_end, NULL ) ;

    gettimeofday(&rgb_split_start, NULL ) ;
    Mat ycbcr_arr[3];
    split(mat_src_ycbcr, ycbcr_arr);
    Mat y = ycbcr_arr[0];
    Mat cb = ycbcr_arr[1];
    Mat cr = ycbcr_arr[2];
    gettimeofday(&rgb_split_end, NULL);

    //开启Y线程
    Mat y_output(newRows, newCols, CV_8U);
    Y_process y_process = {&mace_start, &mace_end, &y_input_process_start, &y_input_process_end,
                           &y_output_process_start, &y_output_process_end, y, y_output};

    if(pthread_create(&__pthread_ptr, NULL, YCalculate, (void *)&y_process) < 0){
        LOGD("processImageByYThread: 开启线程失败!! ---------------------------->>> 串行执行");
        YCalculate((void *)&y_process);
        //return;
    }

    gettimeofday(&cbs_start, NULL);
    Mat cb_output(newRows, newCols, CV_8U);
    //callJavaCubicMethod2(env, clazz, cb, cb_output, newCols, newRows); //call java
    resize(cb, cb_output, cb_output.size(),3, 3, INTER_CUBIC);
    gettimeofday(&cbs_end, NULL);


    gettimeofday(&crs_start, NULL);
    Mat cr_output(newRows, newCols, CV_8U);
    resize(cr, cr_output, cr_output.size(), 3, 3, INTER_CUBIC);
    //callJavaCubicMethod2(env, clazz, cr, cr_output, newCols, newRows);
    gettimeofday(&crs_end, NULL);

    thread_event_wait(__thread_event);
    LOGE("thread_event_wait -> %d", __thread_event.value);

    //merge
    //LOGE("merge -> %d %d %d %d %d %d", y_output.size, cb_output.size, cr_output.size, y_output.depth(), cb_output.depth(), cr_output.depth());
    gettimeofday(&yCbcr_merge_start, NULL);
    Mat mat_dst_ycbcr;
    Mat mat_dst_ycbcr_channels[3];
    gettimeofday(&yCbcr_merge_start, NULL);
    mat_dst_ycbcr_channels[0] = y_output;
    mat_dst_ycbcr_channels[1] = cb_output;
    mat_dst_ycbcr_channels[2] = cr_output;
    merge(mat_dst_ycbcr_channels, 3, mat_dst_ycbcr);
    gettimeofday(&yCbcr_merge_end, NULL);

    gettimeofday(&yCbcr2rgb_start, NULL);
    cvtColor(mat_dst_ycbcr, mat_dst_rgb, COLOR_YCrCb2RGB);
    gettimeofday(&yCbcr2rgb_end, NULL);

    gettimeofday(&mat2image_start, NULL);
    MatToBitmap(env, mat_dst_rgb, jdstBitmap);//mat转成化图片
    gettimeofday(&mat2image_end, NULL);

    gettimeofday(&image_handle_end, NULL);

    printDiff("y_thread image process end !!");
}


JNIEXPORT void JNICALL Java_gdut_bsx_tensorflowtraining_OpenCVUtil_processImageByMaceThread(
        JNIEnv *env, jobject clazz, jobject jsrcBitmap, jobject jdstBitmap) {

    gettimeofday(&image_handle_start, NULL) ;

    gettimeofday(&image2mat_start, NULL ) ;
    Mat mat_src_rgb;
    BitmapToMat(env, jsrcBitmap, mat_src_rgb);//图片转化成mat
    int newRows = mat_src_rgb.rows * 3;
    int newCols = mat_src_rgb.cols * 3;
    gettimeofday(&image2mat_end, NULL ) ;

    gettimeofday(&rgb2yCbcr_start, NULL ) ;
    Mat mat_dst_rgb;
    Mat mat_src_ycbcr;
    cvtColor(mat_src_rgb, mat_src_ycbcr, COLOR_RGB2YCrCb);
    gettimeofday(&rgb2yCbcr_end, NULL ) ;

    gettimeofday(&rgb_split_start, NULL ) ;
    Mat ycbcr_arr[3];
    split(mat_src_ycbcr, ycbcr_arr);
    Mat y = ycbcr_arr[0];
    Mat cb = ycbcr_arr[1];
    Mat cr = ycbcr_arr[2];
    gettimeofday(&rgb_split_end, NULL);

    gettimeofday(&y_input_process_start, NULL ) ;
    float alpha = 2/255.0f;
    Mat mace_y_input(y.rows, y.cols, CV_32F);
    Mat mace_y_output(newRows, newCols, CV_32F);
    //Mat mace_y_output;
    //cvConvertScale(&y, &mace_y_input, alpha, -1);
    y.convertTo(mace_y_input, CV_32F, alpha, -1);
    gettimeofday(&y_input_process_end, NULL ) ;

    //开启mace线程
    MACE_process mace_process = {&mace_start, &mace_end, mace_y_input, mace_y_output};
    if(pthread_create(&__pthread_ptr, NULL, maceCalculate, (void *)&mace_process) < 0){
        LOGD("processImageByMaceThread: 开启线程失败!! ---------------------------->>> 串行执行");
        maceCalculate((void *)&mace_process);
        //return;
    }

    gettimeofday(&cbs_start, NULL);
    Mat cb_output(newRows, newCols, CV_8U);
    //callJavaCubicMethod2(env, clazz, cb, cb_output, newCols, newRows); //call java
    resize(cb, cb_output, cb_output.size(),3, 3, INTER_CUBIC);
    gettimeofday(&cbs_end, NULL);


    gettimeofday(&crs_start, NULL);
    Mat cr_output(newRows, newCols, CV_8U);
    resize(cr, cr_output, cr_output.size(), 3, 3, INTER_CUBIC);
    //callJavaCubicMethod2(env, clazz, cr, cr_output, newCols, newRows);
    gettimeofday(&crs_end, NULL);

    thread_event_wait(__thread_event);
    LOGE("thread_event_wait -> %d", __thread_event.value);

    mace_y_output = mace_process.output_mat; // mace_y_output empty if not set ???
    gettimeofday(&y_output_process_start, NULL);
    Mat y_output(newRows, newCols, CV_8U);
    float beta = 255/2.0f;
    mace_y_output.convertTo(y_output, CV_8U, beta, beta);
    gettimeofday(&y_output_process_end, NULL);

    //merge
    gettimeofday(&yCbcr_merge_start, NULL);
    Mat mat_dst_ycbcr;
    Mat mat_dst_ycbcr_channels[3];
    gettimeofday(&yCbcr_merge_start, NULL);
    mat_dst_ycbcr_channels[0] = y_output;
    mat_dst_ycbcr_channels[1] = cb_output;
    mat_dst_ycbcr_channels[2] = cr_output;
    merge(mat_dst_ycbcr_channels, 3, mat_dst_ycbcr);
    gettimeofday(&yCbcr_merge_end, NULL);

    gettimeofday(&yCbcr2rgb_start, NULL);
    cvtColor(mat_dst_ycbcr, mat_dst_rgb, COLOR_YCrCb2RGB);
    gettimeofday(&yCbcr2rgb_end, NULL);

    gettimeofday(&mat2image_start, NULL);
    MatToBitmap(env, mat_dst_rgb, jdstBitmap);//mat转成化图片
    gettimeofday(&mat2image_end, NULL);

    gettimeofday(&image_handle_end, NULL);

    printDiff("mace_thread image process end !!");
}


JNIEXPORT jint JNICALL Java_gdut_bsx_tensorflowtraining_OpenCVUtil_test(
        JNIEnv *env) {
    return 200;
}

#ifdef __cplusplus
}
#endif