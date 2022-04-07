package sample.util;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.util.*;

public class CvService {

    // Выделение границ
    public void extractBoarders(Mat img) {
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(imgGray, "GRAY");
        Mat edges = new Mat();
        Imgproc.Canny(imgGray, edges, 80, 200);
        CvUtilsFX.showImage(edges, "Canny");
        Mat img3 = new Mat();
        Imgproc.threshold(imgGray, img3, 100, 255,
                Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        Mat edges2 = new Mat();
        Imgproc.Canny(img3, edges2, 80, 200);
        CvUtilsFX.showImage(edges2, "Canny + THRESH_OTSU");
        Mat img4 = new Mat();
        Imgproc.adaptiveThreshold(imgGray, img4, 255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY, 3, 5);
        Mat edges3 = new Mat();
        Imgproc.Canny(img4, edges3, 80, 200);
        CvUtilsFX.showImage(edges3, "Canny + adaptiveThreshold");
        img.release();
        img3.release();
        img4.release();
        imgGray.release();
        edges.release();
        edges2.release();
        edges3.release();
    }

    // Поиск контуров
    public void extractCircuit(Mat img) {
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
        Mat edges = new Mat();
        Imgproc.Canny(imgGray, edges, 80, 200);
        CvUtilsFX.showImage(edges, "Canny");
        Mat edgesCopy = edges.clone(); // Создаем копию
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edgesCopy, contours, hierarchy,
                Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_SIMPLE);
        //System.out.println(contours.size());
        //System.out.println(hierarchy.size());
        //System.out.println(hierarchy.dump());
        Imgproc.drawContours(img, contours, -1, CvUtils.COLOR_RED);
        CvUtilsFX.showImage(img, "drawContours");
        img.release();
        imgGray.release();
        edges.release();
        edgesCopy.release();
        hierarchy.release();
    }


    // Вычисление площади контура и обводка контура рамкой
    public void extractCircuitAndSurround(Mat img) {
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
        Mat edges = new Mat();
        Imgproc.Canny(imgGray, edges, 80, 200);
        Mat edgesCopy = edges.clone(); // Создаем копию
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(edgesCopy, contours, new Mat(),
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);
        for (int i = 0, j = contours.size(); i < j; i++) {
            //System.out.println(Imgproc.contourArea(contours.get(i)));
            Rect r = Imgproc.boundingRect(contours.get(i));
            //System.out.println("boundingRect = " + r);
            double len = Imgproc.arcLength(
                    new MatOfPoint2f(contours.get(i).toArray()), true);
            //System.out.println("arcLength = " + len);
            Imgproc.rectangle(img, new Point(r.x, r.y),
                    new Point(r.x + r.width - 1, r.y + r.height - 1),
                    CvUtils.COLOR_BLUE);
        }
        CvUtilsFX.showImage(img, "boundingRect");
        img.release();
        imgGray.release();
        edges.release();
        edgesCopy.release();
    }

    // Вычисление моментов и центров масс
    public void countMomentsAndMassCenters(Mat img) {
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
        Mat edges = new Mat();
        Imgproc.Canny(imgGray, edges, 80, 200);
        Mat edgesCopy = edges.clone(); // Создаем копию
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(edgesCopy, contours, new Mat(),
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);
        for (int i = 0, j = contours.size(); i < j; i++) {
            Rect r = Imgproc.boundingRect(contours.get(i));
            Moments m = Imgproc.moments(contours.get(i));
//            // Пространственные моменты
//            System.out.println("get_m00 = " + m.get_m00());
//            System.out.println("get_m01 = " + m.get_m01());
//            System.out.println("get_m02 = " + m.get_m02());
//            System.out.println("get_m03 = " + m.get_m03());
//            System.out.println("get_m10 = " + m.get_m10());
//            System.out.println("get_m11 = " + m.get_m11());
//            System.out.println("get_m12 = " + m.get_m12());
//            System.out.println("get_m20 = " + m.get_m20());
//            System.out.println("get_m21 = " + m.get_m21());
//            System.out.println("get_m30 = " + m.get_m30());
//            // Центральные моменты
//            System.out.println("get_mu02 = " + m.get_mu02());
//            System.out.println("get_mu03 = " + m.get_mu03());
//            System.out.println("get_mu11 = " + m.get_mu11());
//            System.out.println("get_mu12 = " + m.get_mu12());
//            System.out.println("get_mu20 = " + m.get_mu20());
//            System.out.println("get_mu21 = " + m.get_mu21());
//            System.out.println("get_mu30 = " + m.get_mu30());
//            // Нормализованные центральные моменты
//            System.out.println("get_nu02 = " + m.get_nu02());
//            System.out.println("get_nu03 = " + m.get_nu03());
//            System.out.println("get_nu11 = " + m.get_nu11());
//            System.out.println("get_nu12 = " + m.get_nu12());
//            System.out.println("get_nu20 = " + m.get_nu20());
//            System.out.println("get_nu21 = " + m.get_nu21());
//            System.out.println("get_nu30 = " + m.get_nu30());
            // Центр масс
            double x_cm = m.get_m10() / m.get_m00();
            double y_cm = m.get_m01() / m.get_m00();
            Imgproc.circle(img, new Point(x_cm, y_cm), 3, new Scalar(0, 255, 255),
                    Core.FILLED);
// Инвариантные моменты
            Mat hu = new Mat();
            Imgproc.HuMoments(m, hu);
//            System.out.println("HuMoments " + hu.dump());
            Imgproc.rectangle(img, new Point(r.x, r.y),
                    new Point(r.x + r.width - 1, r.y + r.height - 1),
                    CvUtils.COLOR_GREEN);
        }
        CvUtilsFX.showImage(img, "boundingRect");
        img.release();
        imgGray.release();
        edges.release();
        edgesCopy.release();
    }

    // Сравнение контуров
    public void compareCircuits(Mat img) {
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
        Mat edges = new Mat();
        Imgproc.Canny(imgGray, edges, 80, 200);
        Mat edgesCopy = edges.clone(); // Создаем копию
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(edgesCopy, contours, new Mat(),
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint shape = new MatOfPoint();
        if (contours.size() >= 4) {
            shape = contours.get(3);
            Imgproc.drawContours(img, contours, 3, CvUtils.COLOR_BLUE);
        }
        double min = Double.MAX_VALUE, value = 0;
        int index = -1;
        for (int i = 0, j = contours.size(); i < j; i++) {
            value = Imgproc.matchShapes(contours.get(i), shape,
                    Imgproc.CV_CONTOURS_MATCH_I3, 0);
            if (value < min) {
                min = value;
                index = i;
            }
//            System.out.println("CV_CONTOURS_MATCH_I1: " + i + " " +
//                    Imgproc.matchShapes(contours.get(i), shape,
//                            Imgproc.CV_CONTOURS_MATCH_I1, 0));
//            System.out.println("CV_CONTOURS_MATCH_I2: " + i + " " +
//                    Imgproc.matchShapes(contours.get(i), shape,
//                            Imgproc.CV_CONTOURS_MATCH_I2, 0));
//            System.out.println("CV_CONTOURS_MATCH_I3: " + i + " " +
//                    Imgproc.matchShapes(contours.get(i), shape,
//                            Imgproc.CV_CONTOURS_MATCH_I3, 0));
        }
        Rect r = Imgproc.boundingRect(contours.get(index));
        Imgproc.rectangle(img, new Point(r.x, r.y),
                new Point(r.x + r.width - 1, r.y + r.height - 1),
                CvUtils.COLOR_RED);
//        System.out.println("Лучшее совпадение: индекс " + index +
//                " значение " + min);
        CvUtilsFX.showImage(img, "Результат сравнения");
        img.release();
        imgGray.release();
        edges.release();
        edgesCopy.release();
        shape.release();
    }

    // Поиск объекта по цвету
    public void findByColorRange(Mat img, Scalar lowerColor, Scalar upperColor) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV);

        Mat h = new Mat();
        Core.extractChannel(hsv, h, 0);
        Mat img2 = new Mat();

//        Core.inRange(h, new Scalar(40), new Scalar(80), img2);
//        CvUtilsFX.showImage(img2, "Зеленый");
//        Core.inRange(h, new Scalar(100), new Scalar(140), img2);
//        CvUtilsFX.showImage(img2, "Синий");
//        Core.inRange(hsv, new Scalar(0, 200, 200),
//                new Scalar(20, 256, 256), img2);
//        CvUtilsFX.showImage(img2, "Красный");
//        Core.inRange(hsv, new Scalar(0, 0, 0),
//                new Scalar(0, 0, 50), img2);
//        CvUtilsFX.showImage(img2, "Черный");

        Core.inRange(h, lowerColor, upperColor, img2);
        CvUtilsFX.showImage(img2, "Заданный цвет");

        img.release();
        img2.release();
        hsv.release();
        h.release();
    }

    // Поиск отличий (строго статических!)
    public void findDiff(Mat img) {
        if (img.empty()) {
            System.out.println("Не удалось загрузить изображение");
            return;
        }
        CvUtilsFX.showImage(img, "Оригинал");
        Mat img2 = img.clone();
        Imgproc.circle(img2, new Point(100, 100), 20, CvUtils.COLOR_RED,
                Core.FILLED);
        CvUtilsFX.showImage(img2, "Оригинал + круг");
        Mat img3 = new Mat();
        Core.absdiff(img2, img, img3);
        CvUtilsFX.showImage(img3, "Разница");
        Mat img4 = new Mat();
        Imgproc.cvtColor(img3, img4, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(img4, img4, 1, 255, Imgproc.THRESH_BINARY);
        CvUtilsFX.showImage(img4, "threshold");
        img.release();
        img2.release();
        img3.release();
        img4.release();
    }

    // Вычитание фона из текущего кадра
    public void subtractBackground(Mat img) {
        CvUtilsFX.showImage(img, "Оригинал");
        Mat img2 = img.clone();
        Imgproc.circle(img2, new Point(100, 100), 20, CvUtils.COLOR_RED,
                Core.FILLED);
        CvUtilsFX.showImage(img2, "Оригинал + круг");
        BackgroundSubtractor bg = Video.createBackgroundSubtractorMOG2();
        Mat img3 = new Mat();
        bg.apply(img, img3);
        bg.apply(img2, img3);
        CvUtilsFX.showImage(img3, "BackgroundSubtractorMOG2");
        BackgroundSubtractorMOG2 bg2 = Video.createBackgroundSubtractorMOG2();

        img.release();
        img2.release();
        img3.release();
    }

    // Поиск по шаблону
    public void findByTemplate(Mat img, Mat template) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img, "Оригинал");
        Mat img2 = template.clone();
        Imgproc.cvtColor(img2, img2, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img2, "Шаблон");
        Mat result = new Mat();
        Imgproc.matchTemplate(img, img2, result, Imgproc.TM_SQDIFF);
        Core.MinMaxLocResult r = Core.minMaxLoc(result);
        System.out.println(r.minVal + " " + r.minLoc);
        Imgproc.rectangle(img, r.minLoc, new Point(r.minLoc.x + img2.width() - 1,
                r.minLoc.y + img2.height() - 1), CvUtils.COLOR_BLUE);
        CvUtilsFX.showImage(img, "Результат поиска");
        Mat result2 = new Mat();
        Imgproc.matchTemplate(img, img2, result2, Imgproc.TM_CCOEFF);
        Core.MinMaxLocResult r2 = Core.minMaxLoc(result2);
        System.out.println(r2.maxVal + " " + r2.maxLoc);
        img.release();
        img2.release();
        result.release();
        result2.release();
    }

    // Поиск прямых линий
    public void findStraightLines(Mat img) {
        Imgproc.rectangle(img, new Point(20, 20), new Point(120, 70),
                CvUtils.COLOR_GREEN, Core.FILLED);
        Imgproc.line(img, new Point(20, 100), new Point(120, 100),
                CvUtils.COLOR_RED, 1);
        Imgproc.line(img, new Point(20, 120), new Point(120, 120),
                CvUtils.COLOR_RED, 3);
        Imgproc.line(img, new Point(150, 20), new Point(150, 100),
                CvUtils.COLOR_RED, 1);
        Imgproc.line(img, new Point(170, 20), new Point(170, 100),
                CvUtils.COLOR_RED, 3);
        Imgproc.line(img, new Point(200, 20), new Point(280, 100),
                CvUtils.COLOR_RED, 3);
        Imgproc.line(img, new Point(280, 20), new Point(200, 100),
                CvUtils.COLOR_RED, 3);
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
        Mat edges = new Mat();
        Imgproc.Canny(imgGray, edges, 80, 200);
        CvUtilsFX.showImage(edges, "Canny");
        Mat lines = new Mat();
        Imgproc.HoughLinesP(edges, lines, 1, Math.toRadians(2), 20, 30, 0);
        Mat result = new Mat(img.size(), CvType.CV_8UC3, CvUtils.COLOR_WHITE);
        for (int i = 0, r = lines.rows(); i < r; i++) {
            for (int j = 0, c = lines.cols(); j < c; j++) {
                double[] line = lines.get(i, j);
                Imgproc.line(result, new Point(line[0], line[1]),
                        new Point(line[2], line[3]), CvUtils.COLOR_BLACK);
            }
        }
        CvUtilsFX.showImage(result, "Результат");
        img.release();
        imgGray.release();
        edges.release();
        result.release();
    }

    // Поиск кругов
    public void findCircles(Mat img) {
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(imgGray, "Оригинал");
        Mat circles = new Mat();
        Imgproc.HoughCircles(imgGray, circles, Imgproc.HOUGH_GRADIENT,
                2, imgGray.rows() / 4);
        Mat result = new Mat(img.size(), CvType.CV_8UC3, CvUtils.COLOR_WHITE);
        for (int i = 0, r = circles.rows(); i < r; i++) {
            for (int j = 0, c = circles.cols(); j < c; j++) {
                double[] circle = circles.get(i, j);
                Imgproc.circle(result, new Point(circle[0], circle[1]),
                        (int) circle[2], CvUtils.COLOR_BLACK);
            }
        }
        CvUtilsFX.showImage(result, "Результат");
        img.release();
        imgGray.release();
        result.release();
    }

    // Сегментация watershed() -  алгоритм «водораздела»
    public void segmentationWatershed(Mat img) {
        CvUtilsFX.showImage(img, "Оригинал");
// Рисуем маркеры
        Mat mask = new Mat(img.size(), CvType.CV_8UC1, new Scalar(0));
        Imgproc.line(mask, new Point(30, 30), new Point(130, 30),
                new Scalar(255), 5);
        Imgproc.line(mask, new Point(60, 80), new Point(90, 80),
                new Scalar(255), 5);
        Imgproc.line(mask, new Point(180, 80), new Point(210, 80),
                new Scalar(255), 5);
        Imgproc.line(mask, new Point(238, 60), new Point(238, 75),
                new Scalar(255), 5);
        CvUtilsFX.showImage(mask, "Маска с маркерами");
// Находим контуры маркеров
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(mask, contours, new Mat(),
                Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_SIMPLE);
// Отрисовываем контуры разными оттенками серого
        Mat markers = new Mat(img.size(), CvType.CV_32SC1, new Scalar(0));
        int color = 80;
        for (int i = 0, j = contours.size(); i < j; i++) {
            Imgproc.drawContours(markers, contours, i, Scalar.all(color), 1);
            color += 20;
        }
        Imgproc.watershed(img, markers);
// Отображаем результат
        Mat result = new Mat();
        markers.convertTo(result, CvType.CV_8U);
        CvUtilsFX.showImage(result, "Результат");
        img.release();
        mask.release();
        markers.release();
        result.release();
    }

    // Сегментацияметод pyrMeanShiftFiltering() - группирует области
    // с близкими признаками
    public void segmentationPyrMeanShiftFiltering(Mat img) {
        CvUtilsFX.showImage(img, "Оригинал");
        Mat result = new Mat();
        Imgproc.pyrMeanShiftFiltering(img, result, 20, 50, 1,
                new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 5, 1));
        CvUtilsFX.showImage(result, "Результат");
        img.release();
        result.release();
    }

    // Сегментацияметод floodFill() - заливку однородных
    // или градиентных областей каким-либо сплошным цветом
    public void segmentationFloodFill(Mat img) {
        CvUtilsFX.showImage(img, "Оригинал");
        Imgproc.floodFill(img, new Mat(), new Point(20, 20),
                new Scalar(255, 255, 255));
        CvUtilsFX.showImage(img, "Результат");
        img.release();
    }

    // Заливка градиентной области
    public void fillGradient(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img, "Оригинал");
        Mat mask = new Mat(img.rows() + 2, img.cols() + 2,
                CvType.CV_8UC1, new Scalar(0));
        Rect r = new Rect();
        Imgproc.floodFill(img, mask, new Point(0, 0),
                new Scalar(255, 255, 255), r,
                new Scalar(20), new Scalar(40), 4 | (255 << 8));
        CvUtilsFX.showImage(mask, "Маска");
        CvUtilsFX.showImage(img, "Результат");
        System.out.println(r);
        img.release();
        mask.release();
    }

    // Отделить объект от фона
    public void extractObjectFromBackground(Mat img) {
        Mat bgdModel = new Mat();
        Mat fgdModel = new Mat();
// Инициализация с помощью прямоугольной области
        Mat mask = new Mat();
        // Rect rect = new Rect(150, 10, 140, 100);
        Rect rect = new Rect(1, 1, img.width() - 2, img.height() - 2);

        Imgproc.grabCut(img, mask, rect, bgdModel, fgdModel, 1,
                Imgproc.GC_INIT_WITH_RECT);
        /*
        // Инициализация с помощью маски
        Mat mask = new Mat(img.size(), CvType.CV_8UC1,
         new Scalar(Imgproc.GC_BGD));
        Imgproc.rectangle(mask, new Point(1, 1),
         new Point(img.width() - 2, img.height() - 2),
         new Scalar(Imgproc.GC_PR_FGD), Core.FILLED);
        Imgproc.grabCut(img, mask, new Rect(), bgdModel, fgdModel, 1,
         Imgproc.GC_INIT_WITH_MASK);
        */
        // Повторный вызов
        Imgproc.grabCut(img, mask, new Rect(), bgdModel, fgdModel, 1,
                Imgproc.GC_EVAL);

        Mat maskPR_FGD = new Mat();
        Core.compare(mask, new Scalar(Imgproc.GC_PR_FGD), maskPR_FGD,
                Core.CMP_EQ);
        Mat resultPR_FGD = new Mat(img.rows(), img.cols(), CvType.CV_8UC3,
                CvUtils.COLOR_WHITE);
        img.copyTo(resultPR_FGD, maskPR_FGD);
        CvUtilsFX.showImage(resultPR_FGD, "Результат GC_PR_FGD");

        Mat maskPR_BGD = new Mat();
        Core.compare(mask, new Scalar(Imgproc.GC_PR_BGD), maskPR_BGD,
                Core.CMP_EQ);
        Mat resultPR_BGD = new Mat(img.rows(), img.cols(), CvType.CV_8UC3,
                CvUtils.COLOR_WHITE);
        img.copyTo(resultPR_BGD, maskPR_BGD);
        CvUtilsFX.showImage(resultPR_BGD, "Результат GC_PR_BGD");

        img.release();
        mask.release();
        maskPR_BGD.release();
        maskPR_FGD.release();
        resultPR_BGD.release();
        resultPR_FGD.release();
    }

    // Кластеризацию, используя алгоритм K-средних
    public void clusterKAvg(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img, "Оригинал");
        Mat data = img.reshape(1, img.rows() * img.cols() * img.channels());
        data.convertTo(data, CvType.CV_32F, 1.0 / 255);
        Mat bestLabels = new Mat();
        Mat centers = new Mat();
        TermCriteria criteria = new TermCriteria(
                TermCriteria.MAX_ITER + TermCriteria.EPS, 10, 1);
        int K = 3;
        Core.kmeans(data, K, bestLabels, criteria, 5,
                Core.KMEANS_RANDOM_CENTERS, centers);
        Mat colors = new Mat();
        centers.t().convertTo(colors, CvType.CV_8U, 255);
        Mat lut = new Mat(1, 256, CvType.CV_8UC1, new Scalar(0));
        colors.copyTo(
                new Mat(lut, new Range(0, 1), new Range(0, colors.cols())));
        Mat result = bestLabels.reshape(img.channels(), img.rows());
        result.convertTo(result, CvType.CV_8U);
        Core.LUT(result, lut, result);
        CvUtilsFX.showImage(result, "Результат K=" + K);
        img.release();
        data.release();
        result.release();
        bestLabels.release();
        centers.release();
        colors.release();
        lut.release();
    }


    // Детектор углов : cornerHarris()
    public void detectCornerHarris(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img, "Оригинал");
        Mat dst = new Mat();
        Imgproc.cornerHarris(img, dst, 2, 3, 0.04);
        Core.MinMaxLocResult m = Core.minMaxLoc(dst);
        Imgproc.threshold(dst, dst, m.maxVal * 0.01, 1.0, Imgproc.THRESH_BINARY);
        CvUtilsFX.showImage(dst, "Результат");
        img.release();
        dst.release();
    }

    // Детектор углов : cornerMinEigenVal()
    public void detectCornerMinEigenVal(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img, "Оригинал");
        Mat dst = new Mat();
        Imgproc.cornerMinEigenVal(img, dst, 2, 3);
        Mat dst2 = new Mat();
        Imgproc.cornerEigenValsAndVecs(img, dst2, 2, 3);
        Core.MinMaxLocResult m = Core.minMaxLoc(dst);
        System.out.println(dst2.channels()); // 6
        double[] maxDst = dst.get((int) m.maxLoc.y, (int) m.maxLoc.x);
        System.out.println(Arrays.toString(maxDst));
        double[] maxDst2 = dst2.get((int) m.maxLoc.y, (int) m.maxLoc.x);
        System.out.println(Arrays.toString(maxDst2));
        Imgproc.threshold(dst, dst, m.maxVal * 0.01, 1.0, Imgproc.THRESH_BINARY);
        dst.convertTo(dst, CvType.CV_8U, 255);
        CvUtilsFX.showImage(dst, "Результат");
        img.release();
        dst.release();
    }

    // Детектор углов : preCornerDetect()
    public void detectCornerPre(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img, "Оригинал");
        Mat dst = new Mat();
        Imgproc.preCornerDetect(img, dst, 3);
        Core.MinMaxLocResult m = Core.minMaxLoc(dst);
        Imgproc.threshold(dst, dst, m.maxVal * 0.01, 1.0, Imgproc.THRESH_BINARY);
        dst.convertTo(dst, CvType.CV_8U, 255);
        CvUtilsFX.showImage(dst, "Результат");
        img.release();
        dst.release();
    }

    // Детектор сильных углов : goodFeaturesToTrack(
    public void detectCornerGoodFeaturesToTrack(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img, "Оригинал");
        MatOfPoint corners = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(img, corners, 50, 0.01, 10);
        double[] v = null;
        for (int i = 0, r = corners.rows(); i < r; i++) {
            for (int j = 0, c = corners.cols(); j < c; j++) {
                v = corners.get(i, j);
                Imgproc.circle(img, new Point(v[0], v[1]), 2,
                        CvUtils.COLOR_WHITE, Core.FILLED);
            }
        }
        CvUtilsFX.showImage(img, "Результат");
        img.release();
    }

    // Уточнить местоположение углов с субпиксельной точностью
    public void detectCornerSubPix(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        CvUtilsFX.showImage(img, "Оригинал");
        MatOfPoint corners = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(img, corners, 50, 0.01, 10,
                new Mat(), 3, true, 0.04);
        Mat imgCopy = img.clone();
        double[] v = null;
        for (int i = 0, r = corners.rows(); i < r; i++) {
            for (int j = 0, c = corners.cols(); j < c; j++) {
                v = corners.get(i, j);
                Imgproc.circle(imgCopy, new Point(v[0], v[1]), 2,
                        CvUtils.COLOR_WHITE, Core.FILLED);
            }
        }
        CvUtilsFX.showImage(imgCopy, "Результат goodFeaturesToTrack");
// Уточнение положения углов
        MatOfPoint2f corners2 = new MatOfPoint2f(corners.toArray());
        TermCriteria criteria = new TermCriteria(
                TermCriteria.MAX_ITER + TermCriteria.EPS, 100, 0.01);
        Imgproc.cornerSubPix(img, corners2, new Size(5, 5),
                new Size(-1, -1), criteria);
        for (int i = 0, r = corners2.rows(); i < r; i++) {
            for (int j = 0, c = corners2.cols(); j < c; j++) {
                v = corners2.get(i, j);
                Imgproc.circle(img, new Point(v[0], v[1]), 2,
                        CvUtils.COLOR_WHITE, Core.FILLED);
            }
        }
        CvUtilsFX.showImage(img, "Результат cornerSubPix");
        img.release();
        imgCopy.release();
    }

    // Поиск ключевых точек : KeyPoint
    public void keyPointKeyPoint(Mat img) {
        KeyPoint p = new KeyPoint();
        System.out.println(p);
// KeyPoint [pt={0.0, 0.0}, size=0.0, angle=-1.0,
// response=0.0, octave=0, class_id=-1]
        KeyPoint p2 = new KeyPoint(20, 30, 1, 45, 0, 0, -1);
        System.out.println(p2);
// KeyPoint [pt={20.0, 30.0}, size=1.0, angle=45.0,
// response=0.0, octave=0, class_id=-1]
        System.out.println(p2.pt); // {20.0, 30.0}
        System.out.println(p2.size); // 1.0
        System.out.println(p2.angle); // 45.0
        System.out.println(p2.response); // 0.0
        System.out.println(p2.octave); // 0
        System.out.println(p2.class_id);//-1
    }

    // Поиск ключевых точек : MatOfKeyPoin
    public void keyPointMatOfKeyPoint(Mat img) {
        MatOfKeyPoint m = new MatOfKeyPoint(new KeyPoint(20, 30, 1),
                new KeyPoint(10, 80, 2));
        System.out.println(m.dump());
/*
[20, 30, 1, -1, 0, 0, -1;
 10, 80, 2, -1, 0, 0, -1]*/
        System.out.println(m.channels()); // 7
        System.out.println(m.type()); // 53
        System.out.println(m.size()); // 1x2

        ArrayList<KeyPoint> list = new ArrayList<KeyPoint>();
        Collections.addAll(list, new KeyPoint(20, 30, 1),
                new KeyPoint(10, 80, 2));
        MatOfKeyPoint mp = new MatOfKeyPoint();
        mp.fromList(list);
        System.out.println(mp.dump());
/*
[20, 30, 1, -1, 0, 0, -1;
 10, 80, 2, -1, 0, 0, -1]*/
    }

    // Отрисовка ключевых точек :
    public void drawKeyPoint(Mat img) {
        Mat m = new Mat(50, 300, CvType.CV_8UC3, CvUtils.COLOR_WHITE);
        MatOfKeyPoint kp = new MatOfKeyPoint(new KeyPoint(30, 30, 30, 90),
                new KeyPoint(120, 20, 20, 10),
                new KeyPoint(80, 30, 15, 45));
        Mat m2 = new Mat();
        Features2d.drawKeypoints(m, kp, m2);
        CvUtilsFX.showImage(m2, "Без флага");
        Mat m3 = new Mat(50, 300, CvType.CV_8UC3, CvUtils.COLOR_RED);
        Features2d.drawKeypoints(m, kp, m3, CvUtils.COLOR_BLUE,
                Features2d.DRAW_OVER_OUTIMG);
        CvUtilsFX.showImage(m3, "DRAW_OVER_OUTIMG");
        Mat m4 = m.clone();
        Features2d.drawKeypoints(m, kp, m4, Scalar.all(-1),
                Features2d.DRAW_RICH_KEYPOINTS);
        CvUtilsFX.showImage(m4, "DRAW_RICH_KEYPOINTS");
    }


    // FeatureDetector описывает методы поиска ключевых точек на изображении
    // с использованием различных алгоритмов
    public void findKeyPoints(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        MatOfKeyPoint kp = new MatOfKeyPoint();
        FeatureDetector fd = FeatureDetector.create(FeatureDetector.AKAZE);
        fd.detect(img, kp);
        Mat result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE,
                Features2d.DRAW_RICH_KEYPOINTS);
        CvUtilsFX.showImage(result, "AKAZE");
        fd = FeatureDetector.create(FeatureDetector.GFTT);
        kp = new MatOfKeyPoint();
        fd.detect(img, kp);
        result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "GFTT");
        fd = FeatureDetector.create(FeatureDetector.HARRIS);
        kp = new MatOfKeyPoint();
        fd.detect(img, kp);
        result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "HARRIS");
        fd = FeatureDetector.create(FeatureDetector.FAST);
        kp = new MatOfKeyPoint();
        fd.detect(img, kp);
        result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "FAST");
        fd = FeatureDetector.create(FeatureDetector.ORB);
        kp = new MatOfKeyPoint();
        fd.detect(img, kp);
        result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "ORB");
        fd = FeatureDetector.create(FeatureDetector.BRISK);
        kp = new MatOfKeyPoint();
        fd.detect(img, kp);
        result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "BRISK");
        fd = FeatureDetector.create(FeatureDetector.SIMPLEBLOB);
        kp = new MatOfKeyPoint();
        fd.detect(img, kp);
        result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE,
                Features2d.DRAW_RICH_KEYPOINTS);
        CvUtilsFX.showImage(result, "SIMPLEBLOB");
        fd = FeatureDetector.create(FeatureDetector.PYRAMID_MSER);
        kp = new MatOfKeyPoint();
        fd.detect(img, kp);
        result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE,
                Features2d.DRAW_RICH_KEYPOINTS);
        CvUtilsFX.showImage(result, "PYRAMID_MSER");
        img.release();
        result.release();
    }


    // Сравнение ключевых точек : Класс DescriptorExtractor описывает методы для вычисления дескрипторов —
    //векторов, кодирующих геометрию локальной окрестности вокруг ключевой точки.
    public void compareKeyPointsDescriptorExtractor(Mat img) {
        MatOfKeyPoint kp_ORB = new MatOfKeyPoint();
        FeatureDetector fd_ORB = FeatureDetector.create(FeatureDetector.ORB);
        fd_ORB.detect(img, kp_ORB);
        DescriptorExtractor de_ORB = DescriptorExtractor.create(
                DescriptorExtractor.ORB);
        Mat descriptors_ORB = new Mat();
        de_ORB.compute(img, kp_ORB, descriptors_ORB);
        System.out.println(de_ORB.descriptorSize()); // 32
        System.out.println(descriptors_ORB.size()); // 32x262
        System.out.println(descriptors_ORB.channels()); // 1
        Mat result_ORB = new Mat();
        Features2d.drawKeypoints(img, kp_ORB, result_ORB,
                CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result_ORB, "ORB");
        FeatureDetector fd_AKAZE = FeatureDetector.create(FeatureDetector.AKAZE);
        MatOfKeyPoint kp_AKAZE = new MatOfKeyPoint();
        fd_AKAZE.detect(img, kp_AKAZE);
        DescriptorExtractor de_AKAZE = DescriptorExtractor.create(
                DescriptorExtractor.AKAZE);
        Mat descriptors_AKAZE = new Mat();
        de_AKAZE.compute(img, kp_AKAZE, descriptors_AKAZE);
        System.out.println(de_AKAZE.descriptorSize()); // 61
        System.out.println(descriptors_AKAZE.size()); // 61x106
        System.out.println(descriptors_AKAZE.channels()); // 1
        Mat result = new Mat();
        Features2d.drawKeypoints(img, kp_AKAZE, result,
                CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "AKAZE");
        img.release();
        result.release();
    }

    // Сравнение ключевых точек : Класс DMatch описывает результат сравнения дескрипторов.
    public void compareKeyPointsDMatch(Mat img) {
        DMatch m = new DMatch();
        System.out.println(m);
// DMatch [queryIdx=-1, trainIdx=-1, imgIdx=-1, distance=3.4028235E38]
        m.queryIdx = 1;
        m.trainIdx = 1;
        m.imgIdx = 2;
        m.distance = 1.5f;
        System.out.println(m);
// DMatch [queryIdx=1, trainIdx=1, imgIdx=2, distance=1.5]
        DMatch m2 = new DMatch(1, 1, 1.5f);
        System.out.println(m2);
// DMatch [queryIdx=1, trainIdx=1, imgIdx=-1, distance=1.5]
    }

    // Сравнение ключевых точек : Класс MatOfDMatch реализует матрицу с результатами сравнения дескрипторов.
    //Каждый элемент такой матрицы содержит четыре канала.
    public void compareKeyPointsMatOfDMatch(Mat img) {
        MatOfDMatch m = new MatOfDMatch(new DMatch(1, 2, 3f),
                new DMatch(4, 5, 6f));
        System.out.println(m.size()); // 1x2
        System.out.println(m.channels()); // 4
        System.out.println(m.type()); // 29
        System.out.println(CvType.CV_32FC4); // 29
        System.out.println(m.dump());
/*
[1, 2, -1, 3;
 4, 5, -1, 6]*/
    }

    // Сравнение ключевых точек : Класс DescriptorMatcher описывает методы сравнения ключевых точек с использованием различных алгоритмов
    public void compareKeyPointsDescriptorMatcher(Mat img, Mat img2) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(img2, img2, Imgproc.COLOR_BGR2GRAY);
// Находим ключевые точки
        MatOfKeyPoint kp_img = new MatOfKeyPoint();
        MatOfKeyPoint kp_img2 = new MatOfKeyPoint();
        FeatureDetector fd = FeatureDetector.create(FeatureDetector.ORB);
        fd.detect(img, kp_img);
        fd.detect(img2, kp_img2);
// Отрисовываем найденные ключевые точки
        Mat result = new Mat();
        Features2d.drawKeypoints(img, kp_img, result,
                CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "result");
        Mat result2 = new Mat();
        Features2d.drawKeypoints(img2, kp_img2, result2,
                CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result2, "result2");
// Вычисляем дескрипторы
        DescriptorExtractor extractor = DescriptorExtractor.create(
                DescriptorExtractor.ORB);
        Mat descriptors_img = new Mat();
        Mat descriptors_img2 = new Mat();
        extractor.compute(img, kp_img, descriptors_img);
        extractor.compute(img2, kp_img2, descriptors_img2);
// Сравниваем дескрипторы
        MatOfDMatch matches = new MatOfDMatch();
        DescriptorMatcher dm = DescriptorMatcher.create(
                DescriptorMatcher.BRUTEFORCE_HAMMING);
        dm.match(descriptors_img, descriptors_img2, matches);
// Вычисляем минимальное и максимальное значения
        double max_dist = Double.MIN_VALUE, min_dist = Double.MAX_VALUE;
        float dist = 0;
        List<DMatch> list = matches.toList();
        for (int i = 0, j = list.size(); i < j; i++) {
            dist = list.get(i).distance;
            if (dist == 0) continue;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }
        System.out.println("min = " + min_dist + " max = " + max_dist);
// Находим лучшие совпадения
        LinkedList<DMatch> list_good = new LinkedList<DMatch>();
        for (int i = 0, j = list.size(); i < j; i++) {
            if (list.get(i).distance < min_dist * 3) {
                list_good.add(list.get(i));
            }
        }
        System.out.println(list_good.size());
        MatOfDMatch mat_good = new MatOfDMatch();
        mat_good.fromList(list_good);
// Отрисовываем результат
        Mat outImg = new Mat(img.rows() + img2.rows() + 10,
                img.cols() + img2.cols() + 10,
                CvType.CV_8UC3, CvUtils.COLOR_BLACK);
        Features2d.drawMatches(img, kp_img, img2, kp_img2, mat_good, outImg,
                new Scalar(255, 255, 255), Scalar.all(-1), new MatOfByte(),
                Features2d.NOT_DRAW_SINGLE_POINTS);
        CvUtilsFX.showImage(outImg, "Результат сравнения");
        img.release();
        img2.release();
        kp_img.release();
        kp_img2.release();
        descriptors_img.release();
        descriptors_img2.release();
        matches.release();
        mat_good.release();
        result.release();
        result2.release();
        outImg.release();
    }


    // Отрисовка найденных совпадающих ключевых точек
    // import org.opencv.features2d.Features2d;
    // public static void drawMatches
    public void drawSameKeyPoints(Mat img, Mat img1) {
    }

    // Создание панораммы
    public void createPanorama(Mat img_orig, Mat img2_orig) {
        Mat img = new Mat();
        Mat img2 = new Mat();
        Imgproc.cvtColor(img_orig, img, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(img2_orig, img2, Imgproc.COLOR_BGR2GRAY);
// Находим ключевые точки
        MatOfKeyPoint kp_img = new MatOfKeyPoint();
        MatOfKeyPoint kp_img2 = new MatOfKeyPoint();
        FeatureDetector fd = FeatureDetector.create(FeatureDetector.ORB);
        fd.detect(img, kp_img);
        fd.detect(img2, kp_img2);
// Вычисляем дескрипторы
        DescriptorExtractor extractor = DescriptorExtractor.create(
                DescriptorExtractor.ORB);
        Mat descriptors_img = new Mat();
        Mat descriptors_img2 = new Mat();
        extractor.compute(img, kp_img, descriptors_img);
        extractor.compute(img2, kp_img2, descriptors_img2);
// Сравниваем дескрипторы
        MatOfDMatch matches = new MatOfDMatch();
        DescriptorMatcher dm = DescriptorMatcher.create(
                DescriptorMatcher.BRUTEFORCE_HAMMING);
        dm.match(descriptors_img, descriptors_img2, matches);
// Вычисляем минимальное значение
        double min_dist = Double.MAX_VALUE, dist = 0;
        List<DMatch> list = matches.toList();
        for (int i = 0, j = list.size(); i < j; i++) {
            dist = list.get(i).distance;
            if (dist == 0) continue;
            if (dist < min_dist) min_dist = dist;
        }
// Находим лучшие совпадения
        LinkedList<DMatch> list_good = new LinkedList<DMatch>();
        for (int i = 0, j = list.size(); i < j; i++) {
            if (list.get(i).distance < min_dist * 3) {
                list_good.add(list.get(i));
            }
        }
        if (list_good.size() < 4) {
            System.out.println("Мало хороших совпадений: " + list_good.size());
            return;
        }
        MatOfDMatch mat_good = new MatOfDMatch();
        mat_good.fromList(list_good);
// Отрисовываем результат сравнения
        Mat outImg = new Mat(img.rows() + img2.rows() + 10,
                img.cols() + img2.cols() + 10,
                CvType.CV_8UC3, CvUtils.COLOR_WHITE);
        Features2d.drawMatches(img, kp_img, img2, kp_img2, mat_good, outImg,
                new Scalar(255, 255, 255), Scalar.all(-1), new MatOfByte(),
                Features2d.NOT_DRAW_SINGLE_POINTS);
        CvUtilsFX.showImage(outImg, "Результат сравнения");
// Выбираем 50 лучших точек
        list_good.sort(new Comparator<DMatch>() {
            @Override
            public int compare(DMatch obj1, DMatch obj2) {
                if (obj1.distance < obj2.distance) return -1;
                if (obj1.distance > obj2.distance) return 1;
                return 0;
            }
        });
        List<KeyPoint> keys1 = kp_img.toList();
        List<KeyPoint> keys2 = kp_img2.toList();
        LinkedList<Point> list1 = new LinkedList<Point>();
        LinkedList<Point> list2 = new LinkedList<Point>();
        DMatch dmatch = null;
        int count = 50;
        if (list_good.size() < count) {
            count = list_good.size();
        }
        for (int i = 0; i < count; i++) {
            dmatch = list_good.get(i);
            list1.add(keys1.get(dmatch.queryIdx).pt);
            list2.add(keys2.get(dmatch.trainIdx).pt);
        }
        MatOfPoint2f p1 = new MatOfPoint2f();
        MatOfPoint2f p2 = new MatOfPoint2f();
        p1.fromList(list1);
        p2.fromList(list2);
// Вычисляем матрицу трансформации
        Mat h = Calib3d.findHomography(p2, p1, Calib3d.RANSAC, 3);
        if (h.empty()) {
            System.out.println("Не удалось рассчитать матрицу трансформации");
            return;
        }
// Применяем матрицу трансформации
        Mat panorama = new Mat();
        Imgproc.warpPerspective(img2_orig, panorama, h,
                new Size(img_orig.width() + img2_orig.width(),
                        img_orig.height() + img2_orig.height()),
                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT,
                Scalar.all(0));
        CvUtilsFX.showImage(panorama, "Результат трансформации");
        Mat subMat = panorama.submat(
                new Rect(0, 0, img_orig.width(), img_orig.height()));
        img_orig.copyTo(subMat);
        CvUtilsFX.showImage(panorama, "Панорама");
        img.release();
        img2.release();
        img_orig.release();
        img2_orig.release();
        kp_img.release();
        kp_img2.release();
        descriptors_img.release();
        descriptors_img2.release();
        matches.release();
        mat_good.release();
        subMat.release();
        panorama.release();
        outImg.release();
        h.release();
        p1.release();
        p2.release();

    }

    // Класс ORB : Пример создания объекта класса ORB, поиска ключевых точек и их отрисовки
    public void keyPointsORB(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        ORB orb = ORB.create();
        MatOfKeyPoint kp = new MatOfKeyPoint();
        orb.detect(img, kp);
        Mat result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "ORB");
        img.release();
        result.release();

        ORB orb1 = ORB.create();
        MatOfKeyPoint kp1 = new MatOfKeyPoint();
        orb1.detect(img, kp1);
        Mat descriptors = new Mat();
        orb1.compute(img, kp1, descriptors);
        System.out.println(descriptors.size()); // 32x262
        System.out.println(descriptors.channels()); // 1
    }

    // Класс GFTTDetector (тип HARRIS) :  содержат множество методов для получения
    //или изменения настроек. Например, класс GFTTDetector содержит метод
    //setHarrisDetector(), с помощью которого можно выбрать либо тип GFTT, либо
    //HARRIS.
    public void keyPointsGFTTDetector(Mat img) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        GFTTDetector fd = GFTTDetector.create();
        fd.setHarrisDetector(true);
        MatOfKeyPoint kp = new MatOfKeyPoint();
        fd.detect(img, kp);
        Mat result = new Mat();
        Features2d.drawKeypoints(img, kp, result, CvUtils.COLOR_WHITE, 0);
        CvUtilsFX.showImage(result, "GFTTDetector");
        img.release();
        result.release();

        ORB orb = ORB.create();
        MatOfKeyPoint kp1 = new MatOfKeyPoint();
        Mat descriptors = new Mat();
        orb.detectAndCompute(img, new Mat(), kp1, descriptors);
        System.out.println(kp1.size()); // 1x262
        System.out.println(descriptors.size()); // 32x262

    }

//    Выполнить поиск объекта по методу Виолы-Джонса позволяет класс
//    CascadeClassifier. Для работы класса требуется классификатор в формате XML,
//    обученный распознавать какие-либо объекты, — например, лица, глаза и т. п.
//    Найти уже готовые к использованию классификаторы можно в папке sources\data\
//    haarcascades, расположенной в каталоге с установленным дистрибутивом OpenCV.

    // Поиск лица : классификаторы
    // haarcascade_frontalface_alt.xml;
    // haarcascade_frontalface_alt_tree.xml;
    // haarcascade_frontalface_alt2.xml;
    // haarcascade_frontalface_default.xml;
    // haarcascade_profileface.xml.
    public void haarFindFace(Mat img, CascadeClassifier face_detector) {
        MatOfRect faces = new MatOfRect();
        face_detector.detectMultiScale(img, faces);
        for (Rect r : faces.toList()) {
            Imgproc.rectangle(img, new Point(r.x, r.y),
                    new Point(r.x + r.width, r.y + r.height),
                    CvUtils.COLOR_WHITE, 2);
        }
        CvUtilsFX.showImage(img, "Результат");
        img.release();
    }

    // Поиск глаз : классификатор haarcascade_eye.xml.
    public void haarFindEyes(Mat img, CascadeClassifier face_detector, CascadeClassifier eye_detector) {
        MatOfRect faces = new MatOfRect();
        face_detector.detectMultiScale(img, faces);
        for (Rect r : faces.toList()) {
            Mat face = img.submat(r);
            MatOfRect eyes = new MatOfRect();
            eye_detector.detectMultiScale(face, eyes);
            for (Rect r2 : eyes.toList()) {
                Imgproc.rectangle(face, new Point(r2.x, r2.y),
                        new Point(r2.x + r2.width, r2.y + r2.height),
                        CvUtils.COLOR_WHITE, 1);
            }
        }
        CvUtilsFX.showImage(img, "Результат");
        img.release();
    }


    // Поиск улыбка : классификатор haarcascade_smile.xml
    public void haarFindSmile(Mat img, CascadeClassifier face_detector, CascadeClassifier smile_detector) {
        MatOfRect faces = new MatOfRect();
        face_detector.detectMultiScale(img, faces);
        for (Rect r : faces.toList()) {
            Mat face = img.submat(r);
            MatOfRect smile = new MatOfRect();
            smile_detector.detectMultiScale(face, smile);
            for (Rect r3 : smile.toList()) {
                Imgproc.rectangle(face, new Point(r3.x, r3.y),
                        new Point(r3.x + r3.width, r3.y + r3.height),
                        CvUtils.COLOR_WHITE, 1);
            }
        }
        CvUtilsFX.showImage(img, "Результат");
        img.release();
    }

    // Поиск носа : классификатором haarcascade_mcs_nose.xml
    public void haarFindNose(Mat img, CascadeClassifier face_detector, CascadeClassifier nose_detector) {
        MatOfRect faces = new MatOfRect();
        face_detector.detectMultiScale(img, faces);
        for (Rect r : faces.toList()) {
            Mat face = img.submat(r);
            MatOfRect nose = new MatOfRect();
            nose_detector.detectMultiScale(face, nose);
            for (Rect r3 : nose.toList()) {
                Imgproc.rectangle(face, new Point(r3.x, r3.y),
                        new Point(r3.x + r3.width, r3.y + r3.height),
                        CvUtils.COLOR_WHITE, 1);
            }
        }
        CvUtilsFX.showImage(img, "Результат");
        img.release();
    }
}
