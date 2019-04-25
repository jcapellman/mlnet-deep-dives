using System;
using System.IO;

using mlbinaryclassifier.Structures;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using Microsoft.ML;
    
namespace mlbinaryclassifier
{
    public class BCAttrition : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var baseTrainingDataView = MlContext.Data.LoadFromTextFile<BCData>(args[(int)CommandLineArguments.INPUT_FILE], hasHeader: true, separatorChar: ';');
           
            var pipeline = MlContext.Transforms.Text.FeaturizeText("Content", "Features")
                .Append(MlContext.BinaryClassification.Trainers.FastTree(numberOfLeaves: 2, numberOfTrees: 10, minimumExampleCountPerLeaf: 1));

            var trainedModel = pipeline.Fit(baseTrainingDataView);

            using (var fs = File.Create(args[(int)CommandLineArguments.OUTPUT_FILE]))
            {
                MlContext.Model.Save(trainedModel, baseTrainingDataView.Schema, fs);
            }

            Console.WriteLine($"Saved model to {args[(int)CommandLineArguments.OUTPUT_FILE]}");
        }

        protected override void Predict(string[] args)
        {
            var prediction = Predictor.Predict<BCData, BCPrediction>(MlContext, args[(int)CommandLineArguments.INPUT_FILE], args[(int)CommandLineArguments.OUTPUT_FILE]);

            Console.WriteLine($"Likely not happy: {prediction.Probability * 100:##.#}%");
        }
    }
}