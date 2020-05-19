import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CNNMain {
    public static void main(String[] args) throws IOException, InterruptedException {
        long seed = 1234;
        long width = 28;
        long height = 28;
        long depth = 1;
        int batchSize = 54;
        int outputSize = 10;
        double learningRate = 0.001;
        int nbrEpoch = 1;
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .list()
                .setInputType(InputType.convolutionalFlat(height, width, depth))
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(depth)
                        .nOut(20) //number of filters
                        .activation(Activation.RELU)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .build())
                .layer(1, new SubsamplingLayer.Builder() //zoom en gardant les details qu'on souhaite
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .build())
                .layer(3, new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(multiLayerConfiguration);
        model.init();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        String path = System.getProperty("user.home") + "/mnist_png";
        File trainFolder = new File(path + "/training");
        FileSplit fileSplitTrain = new FileSplit(trainFolder, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RecordReader recordReaderTrain = new ImageRecordReader(height, width, depth, new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSplitTrain);
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, outputSize);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        dataSetIteratorTrain.setPreProcessor(scaler);
//    while(dataSetIteratorTrain.hasNext()){
//        DataSet dataSet = dataSetIteratorTrain.next();
//        INDArray features = dataSet.getFeatures();
//        INDArray labels = dataSet.getLabels();
//        System.out.println(features.shapeInfoToString());
//        System.out.println(labels.shapeInfoToString());
//        System.out.println("-------------------------------------------------");
//    }
        System.out.println("--------------- Model TRAINING ---------------------");
        for (int i = 0; i < nbrEpoch; i++) {
            System.out.println("Epoque " + i);
            model.fit(dataSetIteratorTrain);
        }
        System.out.println("-------------- MODEL EVALUATION --------------------");
        File evalFolder = new File(path + "/testing");
        FileSplit fileSplitEval = new FileSplit(evalFolder, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RecordReader recordReaderEval = new ImageRecordReader(height, width, depth, new ParentPathLabelGenerator());
        recordReaderEval.initialize(fileSplitEval);
        DataSetIterator dataSetIteratorEval = new RecordReaderDataSetIterator(recordReaderEval, batchSize, 1, outputSize);
        dataSetIteratorEval.setPreProcessor(scaler);
        Evaluation evaluation = new Evaluation();
        while (dataSetIteratorEval.hasNext()){
                DataSet dataSet = dataSetIteratorEval.next();
                INDArray features = dataSet.getFeatures();
                INDArray targetLabels = dataSet.getLabels();
                INDArray predictedLabels = model.output(features);
                evaluation.eval(predictedLabels,targetLabels);
        }
        System.out.println("--------------------- RESULTAT -----------------------");
        System.out.println(evaluation.stats());
    }
}
