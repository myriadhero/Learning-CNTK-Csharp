﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;

namespace console
{
    class Program
    {
        static void Main(string[] args)
        {
            var device = DeviceDescriptor.CPUDevice;


            int numOutputClasses = 2;
            int inputDim = 6000;
            int batchSize = 20; // not sure how to make this increase to 100% of the file, for now

            // full path works "C:\\...
            string dataPath_model = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\mModelZ1.dnn";
            string dataPath_train = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\YXFFData6001Train.txt";
            string dataPath_test = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\YXFFData6001Test.txt";

            // load saved model
            Function model = Function.Load(dataPath_model, device);

            // the model output needs to be processed still
            // out = C.softmax(z)
            var modelOut = CNTKLib.Softmax(model);

            string featureStreamName = "features";
            string labelsStreamName = "labels";

            var streamConfig = new StreamConfiguration[]{
                    new StreamConfiguration(featureStreamName, inputDim),
                    new StreamConfiguration(labelsStreamName, numOutputClasses)
            };

            
            var minibatchSource_train = MinibatchSource.TextFormatMinibatchSource(
                        dataPath_train, streamConfig,
                        MinibatchSource.InfinitelyRepeat, true);

            

            //Variable feature = modelOut.Arguments[0];
            //Variable label = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);


            //define input and output variable and connecting to the stream configuration
            var feature = Variable.InputVariable(new NDShape(1, inputDim), DataType.Float, featureStreamName);
            var label = Variable.InputVariable(new NDShape(1, numOutputClasses), DataType.Float, labelsStreamName);

            ////Step 2: define values, and variables
            //Variable x = Variable.InputVariable(new int[] { 1 }, DataType.Float, "input");
            //Variable y = Variable.InputVariable(new int[] { 1 }, DataType.Float, "output");

            ////Step 2: define training data set from table above
            //var xValues = Value.CreateBatch(new NDShape(1, 1), new float[] { 1f, 2f, 3f, 4f, 5f }, device);
            //var yValues = Value.CreateBatch(new NDShape(1, 1), new float[] { 3f, 5f, 7f, 9f, 11f }, device);

            //var features = Value.CreateBatch(NDShape sampleShape, IEnumerable<T> batch, DeviceDescriptor device);
            //Value.CreateBatch(inputDim,  ,device);


            // prepare the training data
            
            var featureStreamInfo = minibatchSource_train.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource_train.StreamInfo(labelsStreamName);

            var minibatchData = minibatchSource_train.GetNextMinibatch((uint)batchSize, device);

            
            //input
            Variable inputVar = modelOut.Arguments.Single();

            var inputDataMap = new Dictionary<Variable, Value>();
            var inputVal = Value.CreateBatch(new NDShape(1, inputDim),  , device);
            inputDataMap.Add(inputVar, inputVal);

            
            //output
            var outputDataMap = new Dictionary<Variable, Value>();
            Variable outputVar = modelOut.Output;
            outputDataMap.Add(outputVar, null);
            







            // evaluate with loaded data
            modelOut.Evaluate(inputDataMap, outputDataMap,  device);




            //var loss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labelVariable);
            //var evalError = CNTKLib.ClassificationError(classifierOutput, labelVariable);

            Console.Write(" ");

        }
    }
}
