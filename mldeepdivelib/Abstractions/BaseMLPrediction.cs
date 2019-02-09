using System;

using mldeepdivelib.Common;
using mldeepdivelib.Enums;

using Microsoft.ML;

namespace mldeepdivelib.Abstractions
{
    public abstract class BaseMLPrediction
    {
        protected MLContext MlContext;

        protected BaseMLPrediction()
        {
            MlContext = new MLContext(seed: 0);
        }

        protected abstract void Train(string[] args);

        protected abstract void Predict(string[] args);

        protected virtual void FeatureExtraction(string[] args)
        {
            Console.WriteLine("Feature Extraction not implemented");
        }

        private CommandLineResponse parseCommandLine(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine($"No parameters passed to Prediction {System.Environment.NewLine}" +
                                  $"usage: <operation> <arg1> <arg2>");

                return new CommandLineResponse {Success = false};
            }

            if (!Enum.TryParse(typeof(MLOperations), args[(int)CommandLineArguments.OPERATION], out var mlOperation))
            {
                Console.WriteLine($"{args[(int)CommandLineArguments.OPERATION]} is an invalid argument");

                Console.WriteLine("Available Options:");

                Console.WriteLine(string.Join(", ", Enum.GetNames(typeof(MLOperations))));

                return new CommandLineResponse { Success = false };
            }

            if (args.Length >= 3)
            {
                return new CommandLineResponse
                {
                    Success = true,
                    MLOperation = (MLOperations) mlOperation
                };
            }

            Console.WriteLine("Each Operation requires 2 additional arguments");
                
            return new CommandLineResponse { Success = false };
        }

        public void Run(string[] args)
        {
            Console.Clear();

            var response = parseCommandLine(args);

            if (!response.Success)
            {
                return;
            }

            switch (response.MLOperation)
            {
                case MLOperations.train:
                    Train(args);
                    break;
                case MLOperations.predict:
                    Predict(args);
                    break;
                case MLOperations.featureextraction:
                    FeatureExtraction(args);
                    break;
            }
        }
    }
}