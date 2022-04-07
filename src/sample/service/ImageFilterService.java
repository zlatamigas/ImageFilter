package sample.service;

import org.opencv.core.Mat;

public interface ImageFilterService {
    /**
     * Реализация высокочастотных фильтров (увеличение резкости).
     **/
    Mat highPassFilter(Mat src, Mat mask);


    /**
     * Морфологическая обработка.
     * Есть возможность выбора  структурирующего элемента (либо выбирать из списка, либо задавать форму и размер)
     **/
    Mat morphologicalFilter(Mat src, Mat mask, int operation);
}
