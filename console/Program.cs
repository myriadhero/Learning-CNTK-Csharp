using System;
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


            int numOutputClasses = 2; // need to know these from the start
            uint inputDim = 6000;   // also these
            uint batchSize = 50; // not sure how to make this increase to 100% of the file, for now

            // full path works "C:\\...
            //string CurrentFolder = Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location);
            //string dataPath_test = "Data\\YXFFData6001Test.txt";
            //string dataPath = Path.Combine(CurrentFolder, dataPath_test);

            string dataPath_model = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\mModelZ1.dnn";
            string dataPath_train = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\YXFFData6001Train.txt";
            

            // load saved model
            Function model = Function.Load(dataPath_model, device);

            // the model output needs to be processed still
            // out = C.softmax(z)
            var modelOut = CNTKLib.Softmax(model);

            var feature_fromModel = modelOut.Arguments[0];
            var label_fromModel = modelOut.Output;

            string featureStreamName = "features";
            string labelsStreamName = "labels";

            var streamConfig = new StreamConfigurationVector{
                    new StreamConfiguration(featureStreamName, inputDim),
                    new StreamConfiguration(labelsStreamName, numOutputClasses)
            };

            var deserializerConfig_train = CNTKLib.CTFDeserializer(dataPath_train, streamConfig);


            //StreamConfigurationVector streams = new StreamConfigurationVector
            //{
            //new StreamConfiguration("feature", 100),
            //new StreamConfiguration("label", 10)
            //};
            //var deserializerConfiguration = CNTKLib.CTFDeserializer(ctfFilePath, streams);
            MinibatchSourceConfig MBconfig_train = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfig_train })
            {
                MaxSweeps = 1000,
                randomizationWindowInChunks = 0,
                randomizationWindowInSamples = 100000,
            };


            var MBsource_train = CNTK.CNTKLib.CreateCompositeMinibatchSource(MBconfig_train);

            var featureStreamInfo_train = MBsource_train.StreamInfo(featureStreamName);
            var labelStreamInfo_train = MBsource_train.StreamInfo(labelsStreamName);

            var nextBatch_train = MBsource_train.GetNextMinibatch(batchSize, device);

            var MBdensefeature_train = nextBatch_train[featureStreamInfo_train].data;
            var MBdenseLabel_train = nextBatch_train[labelStreamInfo_train].data.GetDenseData<float>(label_fromModel);


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
            
            //var featureStreamInfo = minibatchSource_train.StreamInfo(featureStreamName);
            //var labelStreamInfo = minibatchSource_train.StreamInfo(labelsStreamName);

            //var minibatchData = minibatchSource_train.GetNextMinibatch((uint)batchSize, device);

            
            //input
            Variable inputVar = modelOut.Arguments.Single();

            var inputDataMap = new Dictionary<Variable, Value>();
            inputDataMap.Add(inputVar, MBdensefeature_train);


            //output
            var outputDataMap = new Dictionary<Variable, Value>();
            Variable outputVar = modelOut.Output;
            outputDataMap.Add(outputVar, null);


            // evaluate with loaded data
            modelOut.Evaluate(inputDataMap, outputDataMap,  device);

            var outputData = outputDataMap[outputVar].GetDenseData<float>(outputVar);

            var actualLabels = outputData.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();
            //var loss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labelVariable);
            //var evalError = CNTKLib.ClassificationError(classifierOutput, labelVariable);
            IList<int> expectedLabels = MBdenseLabel_train.Select(l => l.IndexOf(1.0F)).ToList();

            int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

            int labelsLength = actualLabels.Count;

            string correctness(bool comparison){
                if (comparison) return "Correct prediction";
                else return "Incorrect prediction";
            }

            for(int i = 0; i< labelsLength; i++){
                Console.WriteLine($"{i+1}.\tPredicted value:  {actualLabels[i]};\tExpected value:  {expectedLabels[i]};\t{correctness(actualLabels[i] == expectedLabels[i])}.");

            }

            Console.WriteLine($"Validating Model: Total Samples = {batchSize}, Misclassify Count = {misMatches}.");





            Console.Write("Success");

        }
    }
}
