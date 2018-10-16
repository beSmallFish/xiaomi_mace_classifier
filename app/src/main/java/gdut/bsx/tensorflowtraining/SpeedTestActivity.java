package gdut.bsx.tensorflowtraining;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Looper;
import android.os.MessageQueue;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.xiaomi.mace.AppModel;
import com.xiaomi.mace.InitData;

import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadFactory;

import gdut.bsx.tensorflowtraining.ternsorflow.Classifier;
import gdut.bsx.tensorflowtraining.ternsorflow.TensorFlowSuImageClassifier;

public class SpeedTestActivity extends AppCompatActivity {

    private int mPicNumber = 0;
    private TextView mTvResult;
    private EditText mEditText;
    private Button mBtnStart;
    private Classifier classifier;
    private Executor executor;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private static final String MODEL_FILE = "file:///android_asset/model/srmini_S3D1C3G8G08.pb";
    private static final int INPUT_SIZE = 224;
    private static final int INPUT_SIZE_640 = 640;
    private static final int INPUT_SIZE_360 = 360;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_speed_test);
        mBtnStart = findViewById(R.id.btn_start);
        mEditText = findViewById(R.id.editText);
        mTvResult = findViewById(R.id.tv_result);
        mBtnStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mPicNumber = Integer.valueOf(mEditText.getText().toString());
                //  startImageClassifier();
                startMaceClassifier();
                mBtnStart.setEnabled(false);
                mTvResult.setText("processing...");
            }
        });
        Looper.myQueue().addIdleHandler(idleHandler);
    }

    private Bitmap getImageFromAssetsFile(Context context, String fileName) {
        Bitmap image = null;
        AssetManager am = context.getResources().getAssets();
        try {
            InputStream is = am.open(fileName);
            image = BitmapFactory.decodeStream(is);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    /**
     *  主线程消息队列空闲时（视图第一帧绘制完成时）处理耗时事件
     */
    MessageQueue.IdleHandler idleHandler = new MessageQueue.IdleHandler() {
        @Override
        public boolean queueIdle() {

            if (classifier == null) {
                // 创建 Classifier
                classifier = TensorFlowSuImageClassifier.create(SpeedTestActivity.this.getAssets(),
                        MODEL_FILE, INPUT_NAME, OUTPUT_NAME);
            }

            initMace();

            // 初始化线程池
            executor = new ScheduledThreadPoolExecutor(1, new ThreadFactory() {
                @Override
                public Thread newThread(@NonNull Runnable r) {
                    Thread thread = new Thread(r);
                    thread.setDaemon(true);
                    thread.setName("ThreadPool-ImageClassifier");
                    return thread;
                }
            });

            return false;
        }
    };

    private void startImageClassifier() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                long totaltime = 0;
                for (int i = 0; i < mPicNumber; i++) {
                    Bitmap image = getImageFromAssetsFile(SpeedTestActivity.this, "images/"+ i +".png");
                    long start = System.currentTimeMillis();
                    classifier.suImage(image);
                    long end = System.currentTimeMillis();
                    totaltime = end - start + totaltime;
                }
                mBtnStart.post(new Runnable() {
                    @Override
                    public void run() {
                        mBtnStart.setEnabled(true);
                    }
                });
                final long finalTotaltime = totaltime;
                mTvResult.post(new Runnable() {
                    @Override
                    public void run() {
                        mTvResult.setText(finalTotaltime + "ms");
                    }
                });
            }
        });
    }

    private void startMaceClassifier() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    long totaltime = 0;
                    for (int i = 0; i < mPicNumber; i++) {
                        Bitmap image = getImageFromAssetsFile(SpeedTestActivity.this, "images/"+ i +".png");
                        //Bitmap croppedBitmap = BitmapUtils.getScaleBitmap(image, INPUT_SIZE_640, INPUT_SIZE_360);
                        long start = System.currentTimeMillis();
                       // classifier.maceClassifierImage(croppedBitmap);
                        classifier.maceClassifierImage_yCbCr_opencv(image);
                        long end = System.currentTimeMillis();
                        totaltime = end - start + totaltime;
                    }
                    mBtnStart.post(new Runnable() {
                        @Override
                        public void run() {
                            mBtnStart.setEnabled(true);
                        }
                    });
                    final long finalTotaltime = totaltime;
                    mTvResult.post(new Runnable() {
                        @Override
                        public void run() {
                            mTvResult.setText(finalTotaltime + "ms");
                        }
                    });
                } catch (Exception e) {
                    Toast.makeText(SpeedTestActivity.this, "parser error....", Toast.LENGTH_LONG );
                    e.printStackTrace();
                }

            }
        });
    }

    private void initMace(){
        InitData initData = new InitData();
        AppModel.instance.maceMobilenetCreateGPUContext(initData);
        AppModel.instance.maceMobilenetCreateEngine(initData, null);
    }


}
