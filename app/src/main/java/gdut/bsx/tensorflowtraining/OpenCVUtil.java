package gdut.bsx.tensorflowtraining;

import android.graphics.Bitmap;
import android.util.Log;

import com.xiaomi.mace.JniMaceUtils;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class OpenCVUtil {

    /* static {
        try {
            System.loadLibrary("native-lib");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }*/

    public static native int test();

    public static native void processImage(Bitmap bitmap, Bitmap dstBitMap);
    public static native void processImageMix(Bitmap bitmap, Bitmap dstBitMap);
    public static native void processImageByCbCrThread(Bitmap bitmap, Bitmap dstBitMap);
    public static native void processImageByYThread(Bitmap bitmap, Bitmap dstBitMap);
    public static native void processImageByMaceThread(Bitmap bitmap, Bitmap dstBitMap);

    public static float[] callMaceProcess(float[] input, int width, int height){
       // float[] input_array = new float[input.rows() * input.cols()];
      //  input.get(0, 0, input_array);
        long start_time_before = System.currentTimeMillis();
        float[] output = JniMaceUtils.maceMobilenetClassify2(input, width, height, 1);
        long start_time_after = System.currentTimeMillis();
        Log.e("test", "mace size : " + output.length + ", mace use : " + (start_time_after - start_time_before));
        //output.put(0, 0, result);
        //Log.e("test", "mace output : " + output.rows());
        return output;
    }

    public static long callMaceProcess2(long input_nativeObj){
        long start_time = System.currentTimeMillis();
        Mat input_mat = new Mat(input_nativeObj);
        long start_time_before_1 = System.currentTimeMillis();
        float[] input = new float[input_mat.rows() * input_mat.cols()];
        long start_time_before_2 = System.currentTimeMillis();
        input_mat.get(0,0, input);
        long start_time_before_3 = System.currentTimeMillis();

        long start_time_before = System.currentTimeMillis();
        float[] output = JniMaceUtils.maceMobilenetClassify2(input, input_mat.cols(), input_mat.rows(), 1);
        long start_time_after = System.currentTimeMillis();

        Mat output_mat = new Mat(input_mat.rows()*3, input_mat.cols()*3, CvType.CV_32F);
        output_mat.put(0,0, output);
        long end_time = System.currentTimeMillis();
        long time_diff = end_time - start_time;
        Log.e("test", "callMaceProcess2 diff : " + time_diff
                + ", before_mace_1:" + (start_time_before_1 - start_time)
                + ", before_mace_2:" + (start_time_before_2 - start_time_before_1)
                + ", before_mace_3:" + (start_time_before_3 - start_time_before_2)
                + ", before_mace:" + (start_time_before - start_time)
                + ", mace:" + (start_time_after - start_time_before)
                + ", after_mace:" + (end_time - start_time_after)
        );
        return output_mat.nativeObj;
    }

    public static long callMaceProcess3(float[] input_arr, int width, int height){
        long start_time = System.currentTimeMillis();
        long start_time_before = System.currentTimeMillis();
        float[] output = JniMaceUtils.maceMobilenetClassify2(input_arr, width/3, height/3, 1);
        long start_time_after = System.currentTimeMillis();

        Mat output_mat = new Mat(height, width, CvType.CV_32F);
        output_mat.put(0,0, output);
        long end_time = System.currentTimeMillis();
        long time_diff = end_time - start_time;
        Log.e("test", "callMaceProcess2 diff : " + time_diff
                + ", before_mace:" + (start_time_before - start_time)
                + ", mace:" + (start_time_after - start_time_before)
                + ", after_mace:" + (end_time - start_time_after)
        );
        return output_mat.nativeObj;
    }

    public static void callProcess(float[] cb_input, float[] cr_input, float[] cb_output, float[] cr_output, int width, int height){
        Mat cb_input_mat = new Mat();
        cb_input_mat.put(0,0, cb_input);
        Mat cb_output_mat = getScaleMatByType(cb_input_mat, height, width, CvType.CV_8U);
        cb_output_mat.get(0,0,cb_output);

        Mat cr_input_mat = new Mat();
        cr_input_mat.put(0,0, cr_input);
        Mat cr_output_mat = getScaleMatByType(cr_input_mat, height, width, CvType.CV_8U);
        cr_output_mat.get(0,0,cr_output);
    }

    public static byte[] callCubicProcess(Mat cb_input_mat, int width, int height){
        //Log.e("test","callCubicProcess start cb_input ---> " + cb_input.length);
       // int x = width/3;
       // int y = height/3;
       /// Mat cb_input_mat = new Mat(x , y, CvType.CV_8U);
       // Log.e("test","callCubicProcess ---------- cb_output ---> x:" + x + ", y:" + y);
      // cb_input_mat.put(0,0, cb_input);
        Log.e("test","callCubicProcess ---------- cb_output ---> 001 depth:" + ((Mat)cb_input_mat).depth() + ", longVal:" + ((Mat)cb_input_mat).nativeObj);
       // Mat mat = new Mat();
       // ((Mat)cb_input_mat).convertTo(mat, CvType.CV_8U);
        Mat cb_output_mat = getScaleMatByType((Mat)cb_input_mat, width, height, CvType.CV_8U);
        Log.e("test","callCubicProcess ---------- cb_output ---> 000 depth:"  + cb_output_mat.depth());
        byte[] cb_output = new byte[height*width];
        cb_output_mat.get(0,0,cb_output);
        Log.e("test","callCubicProcess ---------- cb_output ---> " + cb_output.length + ", cb_output.natveObj:" + cb_output_mat.nativeObj);
        return cb_output;
    }

    public static long callCubicProcess2(long cb_input_nativeObj, int width, int height){
        Mat cb_input_mat = new Mat(cb_input_nativeObj);
        Log.e("test","callCubicProcess ---------- cb_output ---> 001 longVal:" + cb_input_mat.nativeObj + ", cb_input_nativeObj: " + cb_input_nativeObj);
        Mat cb_output_mat = getScaleMatByType(cb_input_mat, width, height, CvType.CV_8U);
        Log.e("test","callCubicProcess ---------- cb_output ---> 000 depth:"  + cb_output_mat.depth() + "cb_output.natveObj:" + cb_output_mat.nativeObj);
        return cb_output_mat.nativeObj;
    }

    public static Bitmap biCubiInterpolation(Bitmap bitmap){
        double scale = 3; // 缩放比例
        Mat ori_mat = new Mat();
        Utils.bitmapToMat(bitmap, ori_mat);
        //Mat img = Highgui.imread("/sdcard/girl.jpg");// 读入图片，将其转换为Mat
        Size dsize = new Size(bitmap.getWidth() * scale, bitmap.getHeight() * scale); // 设置新图片的大小
        Mat img2 = new Mat(dsize, CvType.CV_16S);// 创建一个新的Mat（opencv的矩阵数据类型）
        //resize(srcImg,dstImg,Size(),0.5,0.5,INTER_CUBIC);
        Imgproc.resize(ori_mat, img2, dsize, scale, scale, Imgproc.INTER_CUBIC);
        Bitmap scaleBitmap = Bitmap.createBitmap((int)(bitmap.getWidth()*scale), (int)(bitmap.getHeight()*scale), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img2, scaleBitmap);
       // bitmap.recycle();

        return scaleBitmap;
    }

    public static double[] getScaleMat(double[] input, int srcWidth, int srcHeight, int destWidth, int destHeight){
        Mat input_mat = new Mat(srcHeight, srcWidth,  CvType.CV_64F);
        Mat output_mat = new Mat(destHeight, destWidth, CvType.CV_64F);
        double[] output = new double[destWidth*destHeight];
        double scale_x = destWidth / srcWidth; // 缩放比例
        double scale_y = destHeight / srcHeight; // 缩放比例
        Size dsize = new Size(destWidth, destHeight); // 设置新图片的大小
        input_mat.put(0,0, input);
        Imgproc.resize(input_mat, output_mat, dsize, scale_x, scale_y, Imgproc.INTER_CUBIC);
        int outputLength = output_mat.get(0, 0, output);
       // Log.e("test", "getScaleMat -> outputLength:" + outputLength);
        return output;
    }

    public static Mat getScaleMat(Mat input_mat, int srcWidth, int srcHeight, int destWidth, int destHeight){
        //Mat input_mat = new Mat(srcHeight, srcWidth,  CvType.CV_64F);
        Mat output_mat = new Mat(destHeight, destWidth, CvType.CV_64F);
        double[] output = new double[destWidth*destHeight];
        double scale_x = destWidth / srcWidth; // 缩放比例
        double scale_y = destHeight / srcHeight; // 缩放比例
        Size dsize = new Size(destWidth, destHeight); // 设置新图片的大小
      //  input_mat.put(0,0, input);
        Imgproc.resize(input_mat, output_mat, dsize, scale_x, scale_y, Imgproc.INTER_CUBIC);
       // int outputLength = output_mat.get(0, 0, output);
        // Log.e("test", "getScaleMat -> outputLength:" + outputLength);
        return output_mat;
    }

    public static Mat getScaleMatByType(Mat input_mat, int destWidth, int destHeight, int type){
        //Mat input_mat = new Mat(srcHeight, srcWidth,  CvType.CV_64F);
        Mat output_mat = new Mat(destHeight, destWidth, type);
        Size dsize = new Size(destWidth, destHeight); // 设置新图片的大小
        Imgproc.resize(input_mat, output_mat, dsize, Imgproc.INTER_CUBIC);
        return output_mat;
    }

}
