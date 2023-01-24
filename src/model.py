def create_model():
    # For this notebook, accuracy will be used to evaluate performance.
    METRICS = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]

    # The model consists of:
    # 1. An input layer that represents the 28x28x3 image flatten.
    # 2. A fully connected layer with 64 units activated by a ReLU function.
    # 3. A single-unit readout layer to output real-scores instead of probabilities.
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation=None),
        ]
    )

    # TFCO by default uses hinge loss — and that will also be used in the model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001), loss="hinge", metrics=METRICS
    )
    return model


def save_model(model, subdir):
    base_dir = tempfile.mkdtemp(prefix="saved_models")
    model_location = os.path.join(base_dir, subdir)
    model.save(model_location, save_format="tf")
    return model_location


def get_eval_results(model_location, eval_subdir):
    base_dir = tempfile.mkdtemp(prefix="saved_eval_results")
    tfma_eval_result_path = os.path.join(base_dir, eval_subdir)

    eval_config_pbtxt = """
        model_specs {
          label_key: "%s"
        }
        metrics_specs {
          metrics {
            class_name: "FairnessIndicators"
            config: '{ "thresholds": [0.22, 0.5, 0.75] }'
          }
          metrics {
            class_name: "ExampleCount"
          }
        }
        slicing_specs {}
        slicing_specs { feature_keys: "%s" }
        options {
          compute_confidence_intervals { value: False }
          disabled_outputs{values: "analysis"}
        }
      """ % (
        LABEL_KEY,
        GROUP_KEY,
    )

    eval_config = text_format.Parse(eval_config_pbtxt, tfma.EvalConfig())

    eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=model_location, tags=[tf.saved_model.SERVING]
    )

    schema_pbtxt = """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "%s"
              value {
                dense_tensor {
                  column_name: "%s"
                  shape {
                    dim { size: 28 }
                    dim { size: 28 }
                    dim { size: 3 }
                  }
                }
              }
            }
          }
        }
        feature {
          name: "%s"
          type: FLOAT
        }
        feature {
          name: "%s"
          type: FLOAT
        }
        feature {
          name: "%s"
          type: BYTES
        }
        """ % (
        IMAGE_KEY,
        IMAGE_KEY,
        IMAGE_KEY,
        LABEL_KEY,
        GROUP_KEY,
    )
    schema = text_format.Parse(schema_pbtxt, schema_pb2.Schema())
    coder = tf_example_record.TFExampleBeamRecord(
        physical_format="inmem",
        schema=schema,
        raw_record_column_name=tfma.ARROW_INPUT_COLUMN,
    )
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=coder.ArrowSchema(),
        tensor_representations=coder.TensorRepresentations(),
    )
    # Run the fairness evaluation.
    with beam.Pipeline() as pipeline:
        _ = (
            tfds_as_pcollection(pipeline, "celeb_a", "test")
            | "ExamplesToRecordBatch" >> coder.BeamSource()
            | "ExtractEvaluateAndWriteResults"
            >> tfma.ExtractEvaluateAndWriteResults(
                eval_config=eval_config,
                eval_shared_model=eval_shared_model,
                output_path=tfma_eval_result_path,
                tensor_adapter_config=tensor_adapter_config,
            )
        )
    return tfma.load_eval_result(output_path=tfma_eval_result_path)
