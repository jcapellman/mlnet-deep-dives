using System;
using System.IO;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Common;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using mlregression.Structures;

using Microsoft.ML;
using Microsoft.ML.Transforms.Normalizers;

namespace mlregression
{
    public class SCDAPrediction : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var modelObject = Activator.CreateInstance<EmploymentHistory>();

            var textReader = MlContext.Data.CreateTextLoader(columns: modelObject.ToColumns(), hasHeader: false, separatorChar: ',');

            var baseTrainingDataView = textReader.Read(args[(int)CommandLineArguments.INPUT_FILE]);

            var (name, min, max) = modelObject.GetLabelAttributes();

            var trainingDataView = MlContext.Data.FilterByColumn(baseTrainingDataView, name, min, max);

            var dataProcessPipeline = MlContext.Transforms.CopyColumns(name, "Label")
                .Append(MlContext.Transforms.Categorical.OneHotEncoding("PositionName", "PositionNameEncoded"))
                .Append(MlContext.Transforms.Normalize("IsMarried", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("BSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("MSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("YearsExperience", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("AgeAtHire", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("HasKids", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("WithinMonthOfVesting", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("DeskDecorations", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("LongCommute", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Concatenate("Features", "PositionNameEncoded", "IsMarried", "BSDegree", "MSDegree", "YearsExperience", "AgeAtHire", "HasKids", "WithinMonthOfVesting", "DeskDecorations", "LongCommute"));

            var trainer = MlContext.Regression.Trainers.StochasticDualCoordinateAscent();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = File.Create(args[(int)CommandLineArguments.OUTPUT_FILE]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[(int)CommandLineArguments.OUTPUT_FILE]}");
        }

        protected override void Predict(string[] args)
        {
            var prediction = Predictor.Predict<EmploymentHistory, EmploymentHistoryPrediction>(MlContext, args[(int)CommandLineArguments.INPUT_FILE], args[(int)CommandLineArguments.OUTPUT_FILE]);

            Console.WriteLine($"Predicted Duration (in months): {prediction.DurationInMonths:0.#}");
        }
    }
}