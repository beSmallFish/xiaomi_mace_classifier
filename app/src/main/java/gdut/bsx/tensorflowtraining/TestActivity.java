package gdut.bsx.tensorflowtraining;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Looper;
import android.os.MessageQueue;
import android.provider.MediaStore;
import android.provider.Settings;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.app.AppCompatDelegate;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.xiaomi.mace.AppModel;
import com.xiaomi.mace.InitData;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadFactory;

import gdut.bsx.tensorflowtraining.ternsorflow.Classifier;
import gdut.bsx.tensorflowtraining.ternsorflow.TensorFlowSuImageClassifier;

public class TestActivity extends AppCompatActivity implements View.OnClickListener {

    public static final String TAG = MainActivity.class.getSimpleName();

    private static final int OPEN_SETTING_REQUEST_COED = 110;
    private static final int TAKE_PHOTO_REQUEST_CODE = 120;
    private static final int PICTURE_REQUEST_CODE = 911;

    private static final int PERMISSIONS_REQUEST = 108;
    private static final int CAMERA_PERMISSIONS_REQUEST_CODE = 119;

    private static final String CURRENT_TAKE_PHOTO_URI = "currentTakePhotoUri";

    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private static final String MODEL_FILE = "file:///android_asset/model/srmini_S3D1C3G8G08.pb";

    private Executor executor;
    private Uri currentTakePhotoUri;

    private ImageView ivPicture, ivPicture_after;
    private Classifier classifier;
    private Uri lastUri;

    private static final int INPUT_SIZE = 224;
    private static final int INPUT_SIZE_640 = 640;
    private static final int INPUT_SIZE_360 = 360;

    private String callType = InitData.CALLS[0];
    private Bitmap dstBitmap;


    /* ------------------------------------------ mace --------------------------------*/

    private InitData initData = new InitData();

    /* ------------------------------------------ mace --------------------------------*/


    static {
        AppCompatDelegate.setCompatVectorFromResourcesEnabled(true);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!isTaskRoot()) {
            finish();
        }

        setContentView(R.layout.activity_test);

        findViewById(R.id.iv_choose_picture).setOnClickListener(this);
        findViewById(R.id.iv_take_photo).setOnClickListener(this);

        findViewById(R.id.imageView3).setOnClickListener(this);
        ivPicture = findViewById(R.id.imageView);
        ivPicture_after = findViewById(R.id.imageView2);
        // 避免耗时任务占用 CPU 时间片造成UI绘制卡顿，提升启动页面加载速度
        Looper.myQueue().addIdleHandler(idleHandler);

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onSaveInstanceState(Bundle savedInstanceState) {
        // 防止拍照后无法返回当前 activity 时数据丢失
        savedInstanceState.putParcelable(CURRENT_TAKE_PHOTO_URI, currentTakePhotoUri);
        super.onSaveInstanceState(savedInstanceState);
    }

    @Override
    protected void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        if (savedInstanceState != null) {
            currentTakePhotoUri = savedInstanceState.getParcelable(CURRENT_TAKE_PHOTO_URI);
        }
    }

    /**
     * 主线程消息队列空闲时（视图第一帧绘制完成时）处理耗时事件
     */
    MessageQueue.IdleHandler idleHandler = new MessageQueue.IdleHandler() {
        @Override
        public boolean queueIdle() {

            if (classifier == null) {
                // 创建 Classifier
                classifier = TensorFlowSuImageClassifier.create(TestActivity.this.getAssets(),
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

            // 请求权限
            requestMultiplePermissions();

            return false;
        }
    };

    /**
     * 请求存储和相机权限
     */
    private void requestMultiplePermissions() {

        String storagePermission = Manifest.permission.WRITE_EXTERNAL_STORAGE;
        String cameraPermission = Manifest.permission.CAMERA;

        int hasStoragePermission = ActivityCompat.checkSelfPermission(this, storagePermission);
        int hasCameraPermission = ActivityCompat.checkSelfPermission(this, cameraPermission);

        List<String> permissions = new ArrayList<>();
        if (hasStoragePermission != PackageManager.PERMISSION_GRANTED) {
            permissions.add(storagePermission);
        }

        if (hasCameraPermission != PackageManager.PERMISSION_GRANTED) {
            permissions.add(cameraPermission);
        }

        if (!permissions.isEmpty()) {
            String[] params = permissions.toArray(new String[permissions.size()]);
            ActivityCompat.requestPermissions(this, params, PERMISSIONS_REQUEST);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (Manifest.permission.WRITE_EXTERNAL_STORAGE.equals(permissions[0]) && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                //permission denied 显示对话框告知用户必须打开权限 (storagePermission )
                // Should we show an explanation?
                // 当app完全没有机会被授权的时候，调用shouldShowRequestPermissionRationale() 返回false
                if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                    // 系统弹窗提示授权
                    showNeedStoragePermissionDialog();
                } else {
                    // 已经被禁止的状态，比如用户在权限对话框中选择了"不再显示”，需要自己弹窗解释
                    showMissingStoragePermissionDialog();
                }
            }
        } else if (requestCode == CAMERA_PERMISSIONS_REQUEST_CODE) {
            if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                showNeedCameraPermissionDialog();
            } else {
                openSystemCamera();
            }
        }
    }

    /**
     * 显示缺失权限提示，可再次请求动态权限
     */
    private void showNeedStoragePermissionDialog() {
        new AlertDialog.Builder(this)
                .setTitle("权限获取提示")
                .setMessage("必须要有存储权限才能获取到图片")
                .setNegativeButton("取消", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.cancel();
                    }
                })
                .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        ActivityCompat.requestPermissions(TestActivity.this,
                                new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST);
                    }
                }).setCancelable(false)
                .show();
    }


    /**
     * 显示权限被拒提示，只能进入设置手动改
     */
    private void showMissingStoragePermissionDialog() {
        new AlertDialog.Builder(this)
                .setTitle("权限获取失败")
                .setMessage("必须要有存储权限才能正常运行")
                .setNegativeButton("取消", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        TestActivity.this.finish();
                    }
                })
                .setPositiveButton("去设置", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        startAppSettings();
                    }
                })
                .setCancelable(false)
                .show();
    }

    private void showNeedCameraPermissionDialog() {
        new AlertDialog.Builder(this)
                .setMessage("摄像头权限被关闭，请开启权限后重试")
                .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                })
                .create().show();
    }

    private static final String PACKAGE_URL_SCHEME = "package:";

    /**
     * 启动应用的设置进行授权
     */
    private void startAppSettings() {
        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        intent.setData(Uri.parse(PACKAGE_URL_SCHEME + getPackageName()));
        startActivityForResult(intent, OPEN_SETTING_REQUEST_COED);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.iv_choose_picture:
                choosePicture();
                break;
            case R.id.iv_take_photo:
                takePhoto();
                break;
            case R.id.imageView3:
                Intent intent = new Intent(TestActivity.this, SpeedTestActivity.class);
                startActivity(intent);
                break;
            default:
                break;
        }
    }

    /**
     * 选择一张图片并裁剪获得一个小图
     */
    private void choosePicture() {
        //Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        // Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        // intent.setType("images/*");
        // startActivityForResult(intent, PICTURE_REQUEST_CODE);

        //调用相册
        Intent intent = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICTURE_REQUEST_CODE);
    }

    /**
     * 使用系统相机拍照
     */
    private void takePhoto() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSIONS_REQUEST_CODE);
        } else {
            openSystemCamera();
        }
    }

    /**
     * 打开系统相机
     */
    private void openSystemCamera() {
        //调用系统相机
        Intent takePhotoIntent = new Intent();
        takePhotoIntent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);

        //这句作用是如果没有相机则该应用不会闪退，要是不加这句则当系统没有相机应用的时候该应用会闪退
        if (takePhotoIntent.resolveActivity(getPackageManager()) == null) {
            Toast.makeText(this, "当前系统没有可用的相机应用", Toast.LENGTH_SHORT).show();
            return;
        }

        String fileName = "TF_" + System.currentTimeMillis() + ".jpg";
        File photoFile = new File(FileUtil.getPhotoCacheFolder(), fileName);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            //通过FileProvider创建一个content类型的Uri
            currentTakePhotoUri = FileProvider.getUriForFile(this, "gdut.bsx.tensorflowtraining.fileprovider", photoFile);
            //对目标应用临时授权该 Uri 所代表的文件
            takePhotoIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } else {
            currentTakePhotoUri = Uri.fromFile(photoFile);
        }

        //将拍照结果保存至 outputFile 的Uri中，不保留在相册中
        takePhotoIntent.putExtra(MediaStore.EXTRA_OUTPUT, currentTakePhotoUri);
        startActivityForResult(takePhotoIntent, TAKE_PHOTO_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == PICTURE_REQUEST_CODE) {
                // 处理选择的图片
                handleInputPhoto(data.getData());
            } else if (requestCode == OPEN_SETTING_REQUEST_COED) {
                requestMultiplePermissions();
            } else if (requestCode == TAKE_PHOTO_REQUEST_CODE) {
                // 如果拍照成功，加载图片并识别
                handleInputPhoto(currentTakePhotoUri);
            }
        }
    }

    public void processImageAgain(View view) {
        if (lastUri == null) {
            try {
                System.out.println(OpenCVUtil.test());
            } catch (Exception e) {
                e.printStackTrace();
            }
            Toast.makeText(TestActivity.this, "请先选择图片", Toast.LENGTH_LONG).show();
            return;
        }
        handleInputPhoto(lastUri);
    }

    /**
     * 处理图片
     *
     * @param imageUri
     */
    private void handleInputPhoto(Uri imageUri) {
      /*  // 加载图片
        GlideApp.with(TestActivity.this).asBitmap().listener(new RequestListener<Bitmap>() {

            @Override
            public boolean onLoadFailed(@Nullable GlideException e, Object model, Target<Bitmap> target, boolean isFirstResource) {
                Log.d(TAG,"handleInputPhoto onLoadFailed");
                Toast.makeText(TestActivity.this, "图片加载失败", Toast.LENGTH_SHORT).show();
                return false;
            }

            @Override
            public boolean onResourceReady(final Bitmap resource, Object model, Target<Bitmap> target, DataSource dataSource, boolean isFirstResource) {
                Log.d(TAG,"handleInputPhoto onResourceReady");
               // startImageClassifier(resource);
                startMaceClassifier(resource);
                return false;
            }
        }).load(imageUri).into(ivPicture);*/
        try {
            Bitmap bitmap = BitmapUtils.getBitmap(TestActivity.this.getContentResolver(), imageUri);
            //        Bitmap bitmap = BitmapUtils.getImageFromAssetsFile(TestActivity.this, imageUri.toString());
            startMaceClassifier(bitmap);
            lastUri = imageUri;
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * 开始图片识别匹配
     *
     * @param bitmap
     */
    private void startImageClassifier(final Bitmap bitmap) {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                System.out.println("hahaha startImageClassifier");
                final Bitmap bitmap2 = big(bitmap);

                ivPicture.post(new Runnable() {
                    @Override
                    public void run() {
                        ivPicture.setImageBitmap(bitmap2);
                    }
                });

                final Bitmap bitmap1 = classifier.suImage(bitmap);

                ivPicture_after.post(new Runnable() {
                    @Override
                    public void run() {
                        //ivPicture_after.setImageBitmap(dstBitmap);
                        System.out.println("hahaha i");
                        ivPicture_after.setImageBitmap(bitmap1);
                    }
                });

            }
        });
    }

    private static Bitmap big(Bitmap bitmap) {
        Matrix matrix = new Matrix();
        matrix.postScale(3.0f, 3.0f); //长和宽放大缩小的比例
        Bitmap resizeBmp = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        return resizeBmp;
    }

    private void initMace() {
        AppModel.instance.maceMobilenetCreateGPUContext(initData);
        AppModel.instance.maceMobilenetCreateEngine(initData, null);
    }

    private void startMaceClassifier(final Bitmap bitmap) {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    Log.i(TAG, Thread.currentThread().getName() + " startMaceClassifier");
                    final Bitmap croppedBitmap = getScaleBitmap(bitmap, 640, 360);
                    // final Bitmap croppedBitmap = bitmap;

                    //System.out.println("hahaha startMaceClassifier");
                    // final Bitmap bitmap2 = big(croppedBitmap);
                    ivPicture.post(new Runnable() {
                        @Override
                        public void run() {
                            ivPicture.setImageBitmap(croppedBitmap);
                        }
                    });
                    //  final Bitmap dstBitmap = classifier.maceClassifierImage(croppedBitmap);
                    //  final Bitmap dstBitmap = classifier.maceClassifierImage_yCbCr(croppedBitmap);
                    //   final Bitmap dstBitmap = OpenCVUtil.biCubiInterpolation(croppedBitmap);
                    if (InitData.CALLS[0].equals(callType)) {
                        dstBitmap = classifier.maceClassifierImage_yCbCr_opencv2(croppedBitmap);
                    } else if (InitData.CALLS[1].equals(callType)){
                        dstBitmap = Bitmap.createBitmap(croppedBitmap.getWidth()*3, croppedBitmap.getHeight()*3, Bitmap.Config.ARGB_8888);
                        OpenCVUtil.processImage(croppedBitmap, dstBitmap);
                    } else if (InitData.CALLS[2].equals(callType)){
                        dstBitmap = Bitmap.createBitmap(croppedBitmap.getWidth()*3, croppedBitmap.getHeight()*3, Bitmap.Config.ARGB_8888);
                        OpenCVUtil.processImageMix(croppedBitmap, dstBitmap);
                    } else if(InitData.CALLS[3].equals(callType)){
                        dstBitmap = Bitmap.createBitmap(croppedBitmap.getWidth()*3, croppedBitmap.getHeight()*3, Bitmap.Config.ARGB_8888);
                        OpenCVUtil.processImageByCbCrThread(croppedBitmap, dstBitmap);
                    } else if(InitData.CALLS[4].equals(callType)){
                        dstBitmap = Bitmap.createBitmap(croppedBitmap.getWidth()*3, croppedBitmap.getHeight()*3, Bitmap.Config.ARGB_8888);
                        OpenCVUtil.processImageByYThread(croppedBitmap, dstBitmap);
                    } else if(InitData.CALLS[5].equals(callType)) {
                        dstBitmap = Bitmap.createBitmap(croppedBitmap.getWidth() * 3, croppedBitmap.getHeight() * 3, Bitmap.Config.ARGB_8888);
                        OpenCVUtil.processImageByMaceThread(croppedBitmap, dstBitmap);
                    }

                    if (dstBitmap == null){
                        Log.e("test", "------------ error : dstBitmap is emtpy !! --------------");
                        return;
                    }
                    ivPicture_after.post(new Runnable() {
                        @Override
                        public void run() {
                            //ivPicture_after.setImageBitmap(dstBitmap);
                            //System.out.println("hahaha i");
                            ivPicture_after.setImageBitmap(dstBitmap);
                        }
                    });
                } catch (Exception e) {
                    Log.e(TAG, "startMaceClassifier getScaleBitmap " + e.getMessage());
                    e.printStackTrace();
                }
            }
        });
    }

    /**
     * 对图片进行缩放
     *
     * @param bitmap
     * @param size
     * @return
     * @throws IOException
     */
    private static Bitmap getScaleBitmap(Bitmap bitmap, int size) throws IOException {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scaleWidth = ((float) size) / width;
        float scaleHeight = ((float) size) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
    }

    private static Bitmap getScaleBitmap(Bitmap bitmap, int targetWidth, int targeHeight) throws IOException {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scaleWidth = ((float) targetWidth) / width;
        float scaleHeight = ((float) targeHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
    }


    @Override
    protected void onResume() {
        super.onResume();

    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    /* Now enable camera view to start receiving frames */
                    //mOpenCvCameraView.setOnTouchListener(Puzzle15Activity.this);
                    //  mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public void showPhoneType(final View view) {
        List<String> menus = Arrays.asList(InitData.DEVICES);
        ContextMenuDialog.show(this, menus, new ContextMenuDialog.OnClickItemListener() {
            @Override
            public void onCLickItem(String content) {
                ((Button) view).setText(content);
                initData.setDevice(content);
                AppModel.instance.maceMobilenetCreateEngine(initData, null);
            }
        });
    }

    public void showCallType(final View view) {
        List<String> menus = Arrays.asList(InitData.CALLS);
        ContextMenuDialog.show(this, menus, new ContextMenuDialog.OnClickItemListener() {
            @Override
            public void onCLickItem(String content) {
                ((Button) view).setText(content);
                callType = content;
            }
        });
    }
}
