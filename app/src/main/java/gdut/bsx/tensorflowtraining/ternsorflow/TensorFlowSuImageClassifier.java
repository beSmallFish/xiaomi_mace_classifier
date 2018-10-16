/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package gdut.bsx.tensorflowtraining.ternsorflow;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import com.xiaomi.mace.AppModel;
import com.xiaomi.mace.JniMaceUtils;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

import gdut.bsx.tensorflowtraining.BiCubicInterpolationScale;
import gdut.bsx.tensorflowtraining.OpenCVUtil;
import org.opencv.core.Scalar;

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowSuImageClassifier implements Classifier {
    private static final String TAG = "TensorFlowImageClassifier";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;

    // Config values.
    private String inputName;
    private String outputName;

    private int[] intValues;
    private int[] outIntValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;
    private static final int INPUT_SIZE = 224;
    private TensorFlowSuImageClassifier() {}

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param inputName The label of the image input node.
     * @param outputName The label of the output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String inputName,
            String outputName) {
        TensorFlowSuImageClassifier c = new TensorFlowSuImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        System.out.println("hahaha Classifier TensorFlowInferenceInterface");


        // Pre-allocate buffers.
        c.outputNames = new String[] {outputName};

        return c;
    }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        return null;
    }

    @Override
    public Bitmap suImage(final Bitmap bitmap) {
        int bitmapSize = bitmap.getWidth() * bitmap.getHeight();
        int outBitmapSize = bitmapSize * 3;
        intValues = new int[bitmapSize];
        floatValues = new float[bitmapSize * 3];
        outputs = new float[outBitmapSize * 3 * 3];
        int[] outIntValues = new int[outBitmapSize * 3];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f * 2 - 1;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f * 2 - 1;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f * 2 - 1;
        }
        inferenceInterface.feed(inputName, floatValues, 1, bitmap.getHeight(), bitmap.getWidth(), 3);
        inferenceInterface.run(outputNames, false);
        inferenceInterface.fetch(outputName, outputs);

        for (int i = 0; i < outIntValues.length; ++i) {
            int r = Math.round(((outputs[i * 3] + 1) / 2 * 255));
            if(r < 0){
                r = 0;
            } else if(r >= 255){
                r = 255;
            }
            int g = Math.round(((outputs[i * 3 + 1] + 1) / 2 * 255));
            if(g < 0){
                g = 0;
            } else if(g >= 255){
                g = 255;
            }

            int b = Math.round(((outputs[i * 3 + 2] + 1) / 2 * 255));
            if(b < 0){
                b = 0;
            } else if(b >= 255){
                b = 255;
            }
            outIntValues[i] =
                 /*   0xFF000000
                            | ((int)( ((outputs[i * 3] + 1) / 2 * 255)) << 16)
                            | (((int) ((outputs[i * 3 + 1] + 1) / 2 * 255)) << 8)
                            | ((int) ((outputs[i * 3 + 2] + 1) / 2 * 255));*/
                    0xFF000000 | (r << 16) | ( g << 8) | (b);
        }
        Bitmap newBitmap = Bitmap.createBitmap(outIntValues, 0, bitmap.getWidth() * 3, bitmap.getWidth()*3, bitmap.getHeight()*3 , Bitmap.Config.ARGB_8888);
        return newBitmap;


    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

    @Override
    public Bitmap maceClassifierImage(final Bitmap bitmap) {
        int bitmapSize = bitmap.getWidth() * bitmap.getHeight();
        int outBitmapSize = bitmapSize * 3;
        intValues = new int[bitmapSize];
        floatValues = new float[bitmapSize * 3];
        outputs = new float[outBitmapSize * 3 * 3];
        int[] outIntValues = new int[outBitmapSize * 3];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f * 2 - 1;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f * 2 - 1;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f * 2 - 1;
        }
      //  inferenceInterface.feed(inputName, floatValues, 1, bitmap.getHeight(), bitmap.getWidth(), 3);
     //   inferenceInterface.run(outputNames, false);
      //  inferenceInterface.fetch(outputName, outputs);
        float[] outputs = JniMaceUtils.maceMobilenetClassify2(floatValues, bitmap.getWidth(), bitmap.getHeight(), 3);
        //float[] outputs = JniMaceUtils.maceMobilenetClassify(floatValues);
        if (outputs == null || outputs.length == 0) {
            Log.e("test", "------------> mace output null !!!!!  inputSize : " + floatValues.length);
            return null;
        }

        Log.e("test", "------------> mace outputs : " + outputs.length);
        for (int i = 0; i < outIntValues.length; ++i) {
            int r = Math.round(((outputs[i * 3 + 0] + 1) / 2 * 255));
            if(r < 0){
                r = 0;
            } else if(r >= 255){
                r = 255;
            }
            int g = Math.round(((outputs[i * 3 + 1] + 1) / 2 * 255));
            if(g < 0){
                g = 0;
            } else if(g >= 255){
                g = 255;
            }

            int b = Math.round(((outputs[i * 3 + 2] + 1) / 2 * 255));
            if(b < 0){
                b = 0;
            } else if(b >= 255){
                b = 255;
            }
            outIntValues[i] =
                 /*   0xFF000000
                            | ((int)( ((outputs[i * 3] + 1) / 2 * 255)) << 16)
                            | (((int) ((outputs[i * 3 + 1] + 1) / 2 * 255)) << 8)
                            | ((int) ((outputs[i * 3 + 2] + 1) / 2 * 255));*/
                    0xFF000000 | (r << 16) | ( g << 8) | (b);
        }
        Bitmap newBitmap = Bitmap.createBitmap(outIntValues, 0, bitmap.getWidth() * 3, bitmap.getWidth()*3, bitmap.getHeight()*3 , Bitmap.Config.ARGB_8888);
        return newBitmap;


    }

    public Bitmap maceClassifierImage_yCbCr(final Bitmap bitmap) {
        int bitmapSize = bitmap.getWidth() * bitmap.getHeight();

        float[] Ys = new float[bitmapSize];
        final double[] Cbs = new double[bitmapSize];
        final double[] Crs = new double[bitmapSize];

        int outBitmapSize = bitmapSize * 3;
        intValues = new int[bitmapSize];
        floatValues = new float[bitmapSize * 3];
        outputs = new float[outBitmapSize * 3 * 3];
        int[] outIntValues = new int[outBitmapSize * 3];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f * 2 - 1;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f * 2 - 1;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f * 2 - 1;

            int R = (val >> 16) & 0xFF;
            int G = (val >> 8) & 0xFF;
            int B = val & 0xFF;

            double y =   16   +   (65.738*R + 129.057*G + 25.064*B)  /256;
            double Cb =  128  +   (-37.945*R - 74.494*G + 112.439*B) /256;
            double Cr =  128  +   (112.439*R - 94.154*G - 18.285*B)  /256;

            Ys[i] =  (float) (y / 255.0f * 2 - 1);
            Cbs[i] = Cb;
            Crs[i] = Cr;

          /*  //处理成 [-1~1]
            Ys[i] = (16
                    + (float)( (((val >> 16) & 0xFF)) * 65.738 /256
                    + (((val >> 8) & 0xFF)) * 129.057/256
                    + ((val & 0xFF)) * 25.064 /256 ))
                    / 255.0f * 2 - 1;

            Cbs[i] = 128
                    - ( (((val >> 16) & 0xFF)) * 37.945 /256
                    - (((val >> 8) & 0xFF)) * 74.494/256
                    + ((val & 0xFF)) * 112.439 /256 );

            Crs[i] = 128
                    + ( (((val >> 16) & 0xFF)) * 112.439 /256
                    - (((val >> 8) & 0xFF)) * 94.154/256
                    - ((val & 0xFF)) * 18.285 /256 );*/
        }

       float[] outputs = JniMaceUtils.maceMobilenetClassify2(Ys, bitmap.getWidth(), bitmap.getHeight(), 1);
       // float[] outputs = JniMaceUtils.maceMobilenetClassify(Ys);
        if (outputs == null || outputs.length == 0) {
            Log.e("test", "------------> mace output null !!!!!  inputSize : " + floatValues.length);
            return null;
        }
        Log.e("test", "------------> mace outputs : " + outputs.length);

        //process Y
        int[] Ys_output = new int[outputs.length];
        for(int i=0;i<outputs.length;i++){
            int y = Math.round((outputs[i] + 1) / 2 * 255);
            if (y < 16) {
                y = 16;
            }
            if (y > 235) {
                y = 235;
            }
            Ys_output[i] = y;
        }

      /*  // just copy the same point
        long copy_start_cbs = System.currentTimeMillis();
        double[] Cbs_output = new double[Cbs.length * 9];
        double[] Crs_output = new double[Crs.length * 9];
        int ori_rows = bitmap.getWidth();
        int current_rows = bitmap.getWidth() * 3;
        for(int i=0;i<Cbs_output.length;i++){

            int x = i / current_rows;
            int y = i % current_rows;

            //缩放
            int ori_x = x / 3;
            int ori_y = y / 3;

            int oriIndex = ori_rows * ori_x + ori_y;
            Cbs_output[i] = Cbs[oriIndex];
        }

        long copy_start_crs = System.currentTimeMillis();
        for(int i=0;i<Crs_output.length;i++){
            int x = i / current_rows;
            int y = i % current_rows;

            //缩放
            int ori_x = x / 3;
            int ori_y = y / 3;

            int oriIndex = ori_rows * ori_x + ori_y;
            Crs_output[i] = Crs[oriIndex];
        }
        long copy_end_crs = System.currentTimeMillis();
        Log.e("test", "cbs_time:" + (copy_start_crs - copy_start_cbs)
                + ", crs_time:" + (copy_end_crs - copy_start_crs)
                + ", totalTime:" + (copy_end_crs - copy_start_cbs));
*/


      /*  long cbs_start_time = System.currentTimeMillis();
        Cbs_output = BiCubicInterpolationScale.imgScale_oneArray(Cbs, bitmap.getWidth(), bitmap.getHeight(), bitmap.getWidth()*3, bitmap.getHeight() * 3);
        long cbs_end_time = System.currentTimeMillis();
        Crs_output = BiCubicInterpolationScale.imgScale_oneArray(Crs, bitmap.getWidth(), bitmap.getHeight(), bitmap.getWidth()*3, bitmap.getHeight() * 3);
        long crs_end_time = System.currentTimeMillis();
        Log.e("test", "------------> mace Cbs_output : " + Cbs_output.length  + ", useTime: " + (cbs_end_time - cbs_start_time)
                + ", Crs_output: " + Crs_output.length + ", useTime: " + (crs_end_time - cbs_end_time)
                + ", totalTime: " + (crs_end_time - cbs_start_time));*/


      // opencv
        long cbs_start_time = System.currentTimeMillis();
        double[] Cbs_output = OpenCVUtil.getScaleMat(Cbs, bitmap.getWidth(), bitmap.getHeight(), bitmap.getWidth()*3, bitmap.getHeight() * 3);
        long cbs_end_time = System.currentTimeMillis();
        double[] Crs_output = OpenCVUtil.getScaleMat(Crs, bitmap.getWidth(), bitmap.getHeight(), bitmap.getWidth()*3, bitmap.getHeight() * 3);
        long crs_end_time = System.currentTimeMillis();
        Log.e("test", "------------> mace Cbs_output : " + Cbs_output.length  + ", useTime: " + (cbs_end_time - cbs_start_time)
                + ", Crs_output: " + Crs_output.length + ", useTime: " + (crs_end_time - cbs_end_time)
                + ", totalTime: " + (crs_end_time - cbs_start_time));

        // mutil thread
        //double[] Cbs_output;
      /*  new Thread(new Runnable() {
            @Override
            public void run() {
                Cbs_output = BiCubicInterpolationScale.imgScale3(Cbs, bitmap.getWidth(), bitmap.getHeight(), bitmap.getWidth()*3, bitmap.getHeight() * 3);
            }
        }).start();

        //double[] Crs_output;
        new Thread(new Runnable() {
            @Override
            public void run() {
                Crs_output = BiCubicInterpolationScale.imgScale3(Crs, bitmap.getWidth(), bitmap.getHeight(), bitmap.getWidth()*3, bitmap.getHeight() * 3);
            }
        }).start();*/

    /*long cbs_start_time = System.currentTimeMillis();
     while (Cbs_output == null || Crs_output == null || Cbs_output.length == 0 || Crs.length == 0){
         try {
             Thread.sleep(1000);
         } catch (Exception e) {
            e.printStackTrace();
         }
     }
     long crs_end_time = System.currentTimeMillis();
    Log.e("test", "------------> mace Cbs_output : " + Cbs_output.length
            + ", Crs_output: " + Crs_output.length
            + ", totalTime: " + (crs_end_time - cbs_start_time));*/

        for(int i=0;i < Ys_output.length;i++){
            int y = Ys_output[i];
            double Cb = Cbs_output[i];
            double Cr = Crs_output[i];

            int r = ((int) ( 298.082 * (y - 16)  + 408.583 * (Cr - 128)    )) >> 8;
            int g = ((int) ( 298.082 * (y - 16) + -100.291 * (Cb - 128) + -208.120 * (Cr - 128)    )) >> 8;
            int b = ((int) ( 298.082 * (y - 16) + 516.411 * (Cb - 128)     )) >> 8;

            if(r < 0){
                r = 0;
            } else if(r >= 255){
                r = 255;
            }

            if(g < 0){
                g = 0;
            } else if(g >= 255){
                g = 255;
            }

            if(b < 0){
                b = 0;
            } else if(b >= 255){
                b = 255;
            }
            outIntValues[i] = 0xFF000000 | (r << 16) | ( g << 8) | (b);
        }


        //YCbCr --> RGBW
        Bitmap newBitmap = Bitmap.createBitmap(outIntValues, 0, bitmap.getWidth() * 3, bitmap.getWidth()*3, bitmap.getHeight()*3 , Bitmap.Config.ARGB_8888);

        return newBitmap;


    }

    public Bitmap maceClassifierImage_yCbCr_test(final Bitmap bitmap) {
        int bitmapSize = bitmap.getWidth() * bitmap.getHeight();

        double[] Ys = new double[bitmapSize];
        double[] Cbs = new double[bitmapSize];
        double[] Crs = new double[bitmapSize];
        int[] testValues = new int[bitmapSize];

        intValues = new int[bitmapSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
/*

            Ys[i] = (16
                    + (float)( (((val >> 16) & 0xFF)) * 65.738 /256
                    + (((val >> 8) & 0xFF)) * 129.057/256
                    + ((val & 0xFF)) * 25.064 /256 ));

            Cbs[i] = 128
                    - ( (((val >> 16) & 0xFF)) * 37.945 /256
                    - (((val >> 8) & 0xFF)) * 74.494/256
                    + ((val & 0xFF)) * 112.439 /256 );

            Crs[i] = 128
                    + ( (((val >> 16) & 0xFF)) * 112.439 /256
                    - (((val >> 8) & 0xFF)) * 94.154/256
                    - ((val & 0xFF)) * 18.285 /256 );
*/
            int R = (val >> 16) & 0xFF;
            int G = (val >> 8) & 0xFF;
            int B = val & 0xFF;

            double y =   16   +   (65.738*R + 129.057*G + 25.064*B)  /256;
            double Cb =  128  +   (-37.945*R - 74.494*G + 112.439*B) /256;
            double Cr =  128  +   (112.439*R - 94.154*G - 18.285*B)  /256;

            Ys[i] =  y;
            Cbs[i] = Cb;
            Crs[i] = Cr;
        }

       /* double[] Ys_output = new double[Ys.length * 9];
        double[] Cbs_output = new double[Cbs.length * 9];
        double[] Crs_output = new double[Crs.length * 9];
        for(int i=0;i<Ys.length;i++){
            Ys_output[i*9 + 0] = Ys[i];
            Ys_output[i*9 + 1] = Ys[i];
            Ys_output[i*9 + 2] = Ys[i];
            Ys_output[i*9 + 3] = Ys[i];
            Ys_output[i*9 + 4] = Ys[i];
            Ys_output[i*9 + 5] = Ys[i];
            Ys_output[i*9 + 6] = Ys[i];
            Ys_output[i*9 + 7] = Ys[i];
            Ys_output[i*9 + 8] = Ys[i];
        }
        for(int i=0;i<Cbs.length;i++){
            Cbs_output[i*9 + 0] = Cbs[i];
            Cbs_output[i*9 + 1] = Cbs[i];
            Cbs_output[i*9 + 2] = Cbs[i];
            Cbs_output[i*9 + 3] = Cbs[i];
            Cbs_output[i*9 + 4] = Cbs[i];
            Cbs_output[i*9 + 5] = Cbs[i];
            Cbs_output[i*9 + 6] = Cbs[i];
            Cbs_output[i*9 + 7] = Cbs[i];
            Cbs_output[i*9 + 8] = Cbs[i];
        }
        for(int i=0;i<Crs.length;i++){
            Crs_output[i*9 + 0] = Crs[i];
            Crs_output[i*9 + 1] = Crs[i];
            Crs_output[i*9 + 2] = Crs[i];
            Crs_output[i*9 + 3] = Crs[i];
            Crs_output[i*9 + 4] = Crs[i];
            Crs_output[i*9 + 5] = Crs[i];
            Crs_output[i*9 + 6] = Crs[i];
            Crs_output[i*9 + 7] = Crs[i];
            Crs_output[i*9 + 8] = Crs[i];
        }
*/
        for(int i=0;i<testValues.length;i++){
            double y = Ys[i];
            double Cb = Cbs[i];
            double Cr = Crs[i];

            int r = ((int) ( 298.082 * (y - 16)  + 408.583 * (Cr - 128)    )) >> 8;
            int g = ((int) ( 298.082 * (y - 16) + -100.291 * (Cb - 128) + -208.120 * (Cr - 128)    )) >> 8;
            int b = ((int) ( 298.082 * (y - 16) + 516.411 * (Cb - 128)    )) >> 8;

            /*int r = (int)(1.164*(y-16)+1.596*(Cr-128));
            int g = (int)(1.164*(y-16)-0.392*(Cb-128)-0.813*(Cr-128));
            int b = (int)(1.164*(y-16)+2.017*(Cb-128));*/

            if(r < 0){
                r = 0;
            } else if(r >= 255){
                r = 255;
            }

            if(g < 0){
                g = 0;
            } else if(g >= 255){
                g = 255;
            }

            if(b < 0){
                b = 0;
            } else if(b >= 255){
                b = 255;
            }

            testValues[i] = 0xFF000000 | (r << 16) | ( g << 8) | (b);
        }

        Bitmap newBitmap = Bitmap.createBitmap(testValues, 0, bitmap.getWidth(), bitmap.getWidth(), bitmap.getHeight() , Bitmap.Config.ARGB_8888);
        return newBitmap;


    }

    public void toRGB(int y, int Cb, int Cr, int[] rgb, int rgbs) {
        rgb[0] = ((int) ( 298.082 * (y - 16)   +
                408.583 * (Cr - 128)    )) >> 8;
        rgb[1] = ((int) ( 298.082 * (y - 16)   +
                -100.291 * (Cb - 128) +
                -208.120 * (Cr - 128)    )) >> 8;
        rgb[2] = ((int) ( 298.082 * (y - 16)   +
                516.411 * (Cb - 128)    )) >> 8;

        for (int i=0; i<3; i++) {
            if (rgb[i] > 255)
                rgb[i] = 255;
            else if (rgb[i] < 0)
                rgb[i] = 0;
        }

    }

    public Bitmap maceClassifierImage_yCbCr_opencv(final Bitmap bitmap) {

        long image_handle_start = System.currentTimeMillis();
        int bitmapSize = bitmap.getWidth() * bitmap.getHeight();

        float[] Ys = new float[bitmapSize];
        // final double[] Cbs = new double[bitmapSize];
        // final double[] Crs = new double[bitmapSize];
        double[] Cbs = new double[bitmapSize];
        double[] Crs = new double[bitmapSize];

        int outBitmapSize = bitmapSize * 3;
        intValues = new int[bitmapSize];
        outputs = new float[outBitmapSize * 3 * 3];
        int[] outIntValues = new int[outBitmapSize * 3];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        long rgb2yCbcr_start = System.currentTimeMillis();
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];

            int R = (val >> 16) & 0xFF;
            int G = (val >> 8) & 0xFF;
            int B = val & 0xFF;

            double y =   16   +   (65.738*R + 129.057*G + 25.064*B)  /256;
            double Cb =  128  +   (-37.945*R - 74.494*G + 112.439*B) /256;
            double Cr =  128  +   (112.439*R - 94.154*G - 18.285*B)  /256;

            Ys[i] =  (float) (y / 255.0f * 2 - 1);
            Cbs[i] = Cb;
            Crs[i] = Cr;
        }
        long rgb2yCbcr_end = System.currentTimeMillis();

        //mace process
        Log.e("test", "mace process start !! inputSize: " + Ys.length);
        long mace_start = System.currentTimeMillis();
        float[] outputs = JniMaceUtils.maceMobilenetClassify2(Ys, bitmap.getWidth(), bitmap.getHeight(), 1);
        // float[] outputs = JniMaceUtils.maceMobilenetClassify(Ys);
        long mace_end = System.currentTimeMillis();
        if (outputs == null || outputs.length == 0) {
            Log.e("test", "------------> mace output null !!!!! ");
            return null;
        }
        Log.e("test", "mace process end !! mace outputs: " + outputs.length);

        //process Y
        long y_output_start = System.currentTimeMillis();
        int[] Ys_output = new int[outputs.length];
        for(int i=0;i<outputs.length;i++){
            int y = Math.round((outputs[i] + 1) / 2 * 255);
            if (y < 16) {
                y = 16;
            }
            if (y > 235) {
                y = 235;
            }
            Ys_output[i] = y;
        }
        long y_output_end = System.currentTimeMillis();

        // opencv
        long cbs_start = System.currentTimeMillis();
        double[] Cbs_output = OpenCVUtil.getScaleMat(Cbs, bitmap.getWidth(), bitmap.getHeight(), bitmap.getWidth()*3, bitmap.getHeight() * 3);
        long cbs_end = System.currentTimeMillis();
        long crs_start = System.currentTimeMillis();
        double[] Crs_output = OpenCVUtil.getScaleMat(Crs, bitmap.getWidth(), bitmap.getHeight(), bitmap.getWidth()*3, bitmap.getHeight() * 3);
        long crs_end = System.currentTimeMillis();
        Log.e("test", "------------> mace Cbs_output : " + Cbs_output.length + ", Crs_output: " + Crs_output.length);


        long yCbcr2rgb_start = System.currentTimeMillis();
        for(int i=0;i < Ys_output.length;i++){
            int y = Ys_output[i];
            double Cb = Cbs_output[i];
            double Cr = Crs_output[i];

            int r = ((int) ( 298.082 * (y - 16)  + 408.583 * (Cr - 128)    )) >> 8;
            int g = ((int) ( 298.082 * (y - 16) + -100.291 * (Cb - 128) + -208.120 * (Cr - 128)    )) >> 8;
            int b = ((int) ( 298.082 * (y - 16) + 516.411 * (Cb - 128)     )) >> 8;

            if(r < 0){
                r = 0;
            } else if(r >= 255){
                r = 255;
            }

            if(g < 0){
                g = 0;
            } else if(g >= 255){
                g = 255;
            }

            if(b < 0){
                b = 0;
            } else if(b >= 255){
                b = 255;
            }
            outIntValues[i] = 0xFF000000 | (r << 16) | ( g << 8) | (b);
        }
        long yCbcr2rgb_end = System.currentTimeMillis();

        Bitmap newBitmap = Bitmap.createBitmap(outIntValues, 0, bitmap.getWidth() * 3, bitmap.getWidth()*3, bitmap.getHeight()*3 , Bitmap.Config.ARGB_8888);

        long image_handle_end = System.currentTimeMillis();
        Log.e("test", "image process end !! totalTime:" + (image_handle_end - image_handle_start)
                + ", rgb2yCbCr_time: " + (rgb2yCbcr_end - rgb2yCbcr_start)
                + ", mace_time:" + (mace_end - mace_start)
                + ", Y_output_time:" + ( y_output_end - y_output_start)
                + ", Cb_output_time:" + (cbs_end - cbs_start)
                + ", Cr_output_time:" + (crs_end - crs_start)
                + ", yCbCr2rgb_time:" + (yCbcr2rgb_end - yCbcr2rgb_start));
        return newBitmap;


    }

    public Bitmap maceClassifierImage_yCbCr_opencv2(final Bitmap bitmap) {

        long image_handle_start = System.currentTimeMillis();
        int bitmapSize = bitmap.getWidth() * bitmap.getHeight();


        long bitmap2mat_start = System.currentTimeMillis();
        float[] Ys = new float[bitmapSize];
        int outBitmapSize = bitmapSize * 3;
        intValues = new int[bitmapSize];
        outputs = new float[outBitmapSize * 3 * 3];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        long bitmap2mat_end = System.currentTimeMillis();

        long rgb2yCbcr_start = System.currentTimeMillis();
        Mat input_mat = new Mat();
        Utils.bitmapToMat(bitmap, input_mat);
        Mat ycbcr = new Mat();
        Imgproc.cvtColor(input_mat, ycbcr, Imgproc.COLOR_RGB2YCrCb);
        long rgb2yCbcr_end = System.currentTimeMillis();


        long rgb_split_start = System.currentTimeMillis();
        List<Mat> chan = new Vector<Mat>();
        Core.split(ycbcr, chan);
        //Mat y = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_32F);
        Mat y = chan.get(0);
        Mat cb = chan.get(1);
        Mat cr = chan.get(2);
        Mat process_y_mat = new Mat();
        long rgb_split_end = System.currentTimeMillis();

        //Mat process_y_mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_32F);
        long y_mace_input_start = System.currentTimeMillis();
        float alpha = 2/255.0f;
        y.convertTo(process_y_mat, CvType.CV_32F, alpha , -1);
        long y_mace_input_end = System.currentTimeMillis();

        //mace process
      //  Log.e("test", "mace process start !! inputSize: " + Ys.length);
        long mace_start = System.currentTimeMillis();
        //Mat y_input_mat = new Mat(bitmap.getWidth(), bitmap.getHeight(),  CvType.CV_32F);
        process_y_mat.get(0, 0, Ys);
        float[] outputs = JniMaceUtils.maceMobilenetClassify2(Ys, bitmap.getWidth(), bitmap.getHeight(), 1);
        // float[] outputs = JniMaceUtils.maceMobilenetClassify(Ys);
        long mace_end = System.currentTimeMillis();
        if (outputs == null || outputs.length == 0) {
            Log.e("test", "------------> mace output null !!!!! ");
            return null;
        }
        Log.e("test", "mace process end !! mace outputs: " + outputs.length);

        //process Y
        long y_mace_output_start = System.currentTimeMillis();
       /* int[] Ys_output = new int[outputs.length];
        for(int i=0;i<outputs.length;i++){
            int y_val = Math.round((outputs[i] + 1) / 2 * 255);
            if (y_val < 16) {
                y_val = 16;
            }
            if (y_val > 235) {
                y_val = 235;
            }
            Ys_output[i] = y_val;
        }
        Mat Y_output_mat = new Mat(bitmap.getHeight()*3, bitmap.getWidth()*3, CvType.CV_64F);
        Y_output_mat.put(0,0, Ys_output);*/

        Mat Y_output_mat_tmp = new Mat(bitmap.getHeight()*3, bitmap.getWidth()*3, CvType.CV_32F);
        Mat Y_output_mat = new Mat(bitmap.getHeight()*3, bitmap.getWidth()*3, CvType.CV_8U);
       // Mat Y_output_mat = new Mat(bitmap.getHeight()*3, bitmap.getWidth()*3, CvType.CV_64F);
        Y_output_mat_tmp.put(0, 0, outputs);
        float beta = 255/2.0f;
        Y_output_mat_tmp.convertTo(Y_output_mat, CvType.CV_8U, beta, beta);
       // Y_output_mat.convertTo();
        //inRange(srcGray, Scalar(65, 65, 65), Scalar(95, 95, 95), dstImage);
       /* for (int i=0;i<Y_output_mat.rows();i++) {
            for(int j=0;j<Y_output_mat.cols();j++){
                if(Y_output_mat.get(i, j)[0] < 16){
                    Y_output_mat.get(i, j)[0] = 16;
                    continue;
                }
                if(Y_output_mat.get(i, j)[0] > 235){
                    Y_output_mat.get(i, j)[0] = 235;
                    continue;
                }
            }
        }*/
       // Core.inRange(Y_output_mat, new Scalar(16, 16, 16), new Scalar(235, 235, 235), Y_output_mat);
      //  Core.bitwise_not(Y_output_mat, Y_output_mat);
      //  Log.e("test", "(0,1):" + Y_output_mat.get(0,1)[0]);
        long y_mace_output_end = System.currentTimeMillis();

        // opencv
        long cbs_start = System.currentTimeMillis();
        Mat Cbs_output_mat = OpenCVUtil.getScaleMatByType(cb, bitmap.getWidth()*3, bitmap.getHeight() * 3, CvType.CV_8U);
        long cbs_end = System.currentTimeMillis();
        long crs_start = System.currentTimeMillis();
        Mat Crs_output_mat = OpenCVUtil.getScaleMatByType(cr, bitmap.getWidth()*3, bitmap.getHeight() * 3, CvType.CV_8U);
        long crs_end = System.currentTimeMillis();
        Log.e("test", "------------> mace Cbs_output : " + Cbs_output_mat.size() + ", Crs_output: " + Crs_output_mat.size());

        long yCbcr_merge_start = System.currentTimeMillis();
        //Imgproc.cvtColor(input_mat, ycbcr, Imgproc.COLOR_YCrCb2RGB);
        Mat scaleYcbcr = new Mat();
        Mat scaleRgb = new Mat();
        List<Mat> chan_2 = new Vector<Mat>();
        chan_2.add(0, Y_output_mat);
        chan_2.add(1, Cbs_output_mat);
        chan_2.add(2, Crs_output_mat);
        Core.merge(chan_2, scaleYcbcr);
        long yCbcr_merge_end = System.currentTimeMillis();

        long yCbcr2rgb_start = System.currentTimeMillis();
        Imgproc.cvtColor(scaleYcbcr, scaleRgb, Imgproc.COLOR_YCrCb2RGB);
        long yCbcr2rgb_end = System.currentTimeMillis();


        long mat2Bitmap_start = System.currentTimeMillis();
        Bitmap newBitmap = Bitmap.createBitmap(bitmap.getWidth()*3, bitmap.getHeight()*3, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(scaleRgb, newBitmap);
        //Bitmap newBitmap = Bitmap.createBitmap(outIntValues, 0, bitmap.getWidth() * 3, bitmap.getWidth()*3, bitmap.getHeight()*3 , Bitmap.Config.ARGB_8888);
        long mat2Bitmap_end = System.currentTimeMillis();

        long image_handle_end = System.currentTimeMillis();
        Log.e("test", "java image process end !! totalTime:" + (image_handle_end - image_handle_start)
                + ", bitmap2mat_time: " + (bitmap2mat_end - bitmap2mat_start)
                + ", rgb2yCbCr_time: " + (rgb2yCbcr_end - rgb2yCbcr_start)
                + ", rgb_split_time:" + (rgb_split_end - rgb_split_start)
                + ", y_mace_input_process: " + (y_mace_input_end - y_mace_input_start)
                + ", mace_time:" + (mace_end - mace_start)
                + ", y_mace_output_process: " + (y_mace_output_end - y_mace_output_start)
                + ", Cb_output_time:" + (cbs_end - cbs_start)
                + ", Cr_output_time:" + (crs_end - crs_start)
                + ", yCbCr_merge_time:" + (yCbcr_merge_end - yCbcr_merge_start)
                + ", yCbCr2rgb_time:" + (yCbcr2rgb_end - yCbcr2rgb_start)
                + ", mat2bitmap_time: " + (mat2Bitmap_end - mat2Bitmap_start));
        return newBitmap;


    }

    public Bitmap maceClassifierImage_yCbCr_opencv3(final Bitmap bitmap) {

        long image_handle_start = System.currentTimeMillis();
        int bitmapSize = bitmap.getWidth() * bitmap.getHeight();


        long bitmap2mat_start = System.currentTimeMillis();
        float[] Ys = new float[bitmapSize];
        int outBitmapSize = bitmapSize * 3;
        intValues = new int[bitmapSize];
        outputs = new float[outBitmapSize * 3 * 3];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        long bitmap2mat_end = System.currentTimeMillis();

        long rgb2yCbcr_start = System.currentTimeMillis();
        Mat input_mat = new Mat();
        Utils.bitmapToMat(bitmap, input_mat);
        Mat ycbcr = new Mat();
        Imgproc.cvtColor(input_mat, ycbcr, Imgproc.COLOR_RGB2YCrCb);
        long rgb2yCbcr_end = System.currentTimeMillis();


        long rgb_split_start = System.currentTimeMillis();
        List<Mat> chan = new Vector<Mat>();
        Core.split(ycbcr, chan);
        //Mat y = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_32F);
        Mat y = chan.get(0);
        Mat cb = chan.get(1);
        Mat cr = chan.get(2);
        Mat process_y_mat = new Mat();
        long rgb_split_end = System.currentTimeMillis();

        //Mat process_y_mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_32F);
        long y_mace_input_start = System.currentTimeMillis();
        float alpha = 2/255.0f;
        y.convertTo(process_y_mat, CvType.CV_32F, alpha , -1);
        long y_mace_input_end = System.currentTimeMillis();

        //mace process
        //  Log.e("test", "mace process start !! inputSize: " + Ys.length);
        long mace_start = System.currentTimeMillis();
        //Mat y_input_mat = new Mat(bitmap.getWidth(), bitmap.getHeight(),  CvType.CV_32F);
        process_y_mat.get(0, 0, Ys);
        float[] outputs = JniMaceUtils.maceMobilenetClassify2(Ys, bitmap.getWidth(), bitmap.getHeight(), 1);
        // float[] outputs = JniMaceUtils.maceMobilenetClassify(Ys);
        long mace_end = System.currentTimeMillis();
        if (outputs == null || outputs.length == 0) {
            Log.e("test", "------------> mace output null !!!!! ");
            return null;
        }
        Log.e("test", "mace process end !! mace outputs: " + outputs.length);

        //process Y
        long y_mace_output_start = System.currentTimeMillis();
       /* int[] Ys_output = new int[outputs.length];
        for(int i=0;i<outputs.length;i++){
            int y_val = Math.round((outputs[i] + 1) / 2 * 255);
            if (y_val < 16) {
                y_val = 16;
            }
            if (y_val > 235) {
                y_val = 235;
            }
            Ys_output[i] = y_val;
        }
        Mat Y_output_mat = new Mat(bitmap.getHeight()*3, bitmap.getWidth()*3, CvType.CV_64F);
        Y_output_mat.put(0,0, Ys_output);*/

        Mat Y_output_mat_tmp = new Mat(bitmap.getHeight()*3, bitmap.getWidth()*3, CvType.CV_32F);
        Mat Y_output_mat = new Mat(bitmap.getHeight()*3, bitmap.getWidth()*3, CvType.CV_8U);
        // Mat Y_output_mat = new Mat(bitmap.getHeight()*3, bitmap.getWidth()*3, CvType.CV_64F);
        Y_output_mat_tmp.put(0, 0, outputs);
        float beta = 255/2.0f;
       // Y_output_mat_tmp.convertTo(Y_output_mat, CvType.CV_8U, beta, beta);
        Core.convertScaleAbs(Y_output_mat_tmp, Y_output_mat, beta, beta);

        // Y_output_mat.convertTo();
        //inRange(srcGray, Scalar(65, 65, 65), Scalar(95, 95, 95), dstImage);
       /* for (int i=0;i<Y_output_mat.rows();i++) {
            for(int j=0;j<Y_output_mat.cols();j++){
                if(Y_output_mat.get(i, j)[0] < 16){
                    Y_output_mat.get(i, j)[0] = 16;
                    continue;
                }
                if(Y_output_mat.get(i, j)[0] > 235){
                    Y_output_mat.get(i, j)[0] = 235;
                    continue;
                }
            }
        }*/
        // Core.inRange(Y_output_mat, new Scalar(16, 16, 16), new Scalar(235, 235, 235), Y_output_mat);
        //  Core.bitwise_not(Y_output_mat, Y_output_mat);
        //  Log.e("test", "(0,1):" + Y_output_mat.get(0,1)[0]);
        long y_mace_output_end = System.currentTimeMillis();

        // opencv
        long cbs_start = System.currentTimeMillis();
        Mat Cbs_output_mat = OpenCVUtil.getScaleMatByType(cb, bitmap.getWidth()*3, bitmap.getHeight() * 3, CvType.CV_8U);
        long cbs_end = System.currentTimeMillis();
        long crs_start = System.currentTimeMillis();
        Mat Crs_output_mat = OpenCVUtil.getScaleMatByType(cr, bitmap.getWidth()*3, bitmap.getHeight() * 3, CvType.CV_8U);
        long crs_end = System.currentTimeMillis();
        Log.e("test", "------------> mace Cbs_output : " + Cbs_output_mat.size() + ", Crs_output: " + Crs_output_mat.size());

        long yCbcr_merge_start = System.currentTimeMillis();
        //Imgproc.cvtColor(input_mat, ycbcr, Imgproc.COLOR_YCrCb2RGB);
        Mat scaleYcbcr = new Mat();
        Mat scaleRgb = new Mat();
        List<Mat> chan_2 = new Vector<Mat>();
        chan_2.add(0, Y_output_mat);
        chan_2.add(1, Cbs_output_mat);
        chan_2.add(2, Crs_output_mat);
        Core.merge(chan_2, scaleYcbcr);
        long yCbcr_merge_end = System.currentTimeMillis();

        long yCbcr2rgb_start = System.currentTimeMillis();
        Imgproc.cvtColor(scaleYcbcr, scaleRgb, Imgproc.COLOR_YCrCb2RGB);
        long yCbcr2rgb_end = System.currentTimeMillis();


        long mat2Bitmap_start = System.currentTimeMillis();
        Bitmap newBitmap = Bitmap.createBitmap(bitmap.getWidth()*3, bitmap.getHeight()*3, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(scaleRgb, newBitmap);
        //Bitmap newBitmap = Bitmap.createBitmap(outIntValues, 0, bitmap.getWidth() * 3, bitmap.getWidth()*3, bitmap.getHeight()*3 , Bitmap.Config.ARGB_8888);
        long mat2Bitmap_end = System.currentTimeMillis();

        long image_handle_end = System.currentTimeMillis();
        Log.e("test", "java image process end !! totalTime:" + (image_handle_end - image_handle_start)
                + ", bitmap2mat_time: " + (bitmap2mat_end - bitmap2mat_start)
                + ", rgb2yCbCr_time: " + (rgb2yCbcr_end - rgb2yCbcr_start)
                + ", rgb_split_time:" + (rgb_split_end - rgb_split_start)
                + ", y_mace_input_process: " + (y_mace_input_end - y_mace_input_start)
                + ", mace_time:" + (mace_end - mace_start)
                + ", y_mace_output_process: " + (y_mace_output_end - y_mace_output_start)
                + ", Cb_output_time:" + (cbs_end - cbs_start)
                + ", Cr_output_time:" + (crs_end - crs_start)
                + ", yCbCr_merge_time:" + (yCbcr_merge_end - yCbcr_merge_start)
                + ", yCbCr2rgb_time:" + (yCbcr2rgb_end - yCbcr2rgb_start)
                + ", mat2bitmap_time: " + (mat2Bitmap_end - mat2Bitmap_start));
        return newBitmap;


    }

}