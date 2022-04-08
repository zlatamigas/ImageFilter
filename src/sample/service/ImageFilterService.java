package sample.service;

import org.opencv.core.Mat;

public interface ImageFilterService {

    Mat highPassFilter(Mat src, Mat mask);

    Mat morphologicalFilter(Mat src, Mat mask, int operation);
}
