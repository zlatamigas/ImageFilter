package sample.service.impl;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import sample.service.ImageFilterService;

import static org.opencv.imgproc.Imgproc.filter2D;

public class ImageFilterServiceImpl implements ImageFilterService {
    @Override
    public Mat highPassFilter(Mat src, Mat mask) {

        Mat res = new Mat(src.rows(), src.cols(), src.type());

        filter2D(src, res, -1, mask, new Point(-1, -1), 0.0);

        return res;
    }

    @Override
    public Mat morphologicalFilter(Mat src, Mat mask, int operation) {

        Mat res = new Mat(src.rows(), src.cols(), src.type());

        Imgproc.morphologyEx(src, res, operation, mask);

        return res;
    }
}
