# Meta-learning for experimenting handwritten Nôm-character recognition

This repo includes code for (1) generating character image data and (2) experimenting with three meta-learning algorithms, namely ProtoNet, MAML, and Proto-MAML, on Nôm character recognition.

*A gentle note:* Please run the [`env.sh`](./env.sh) before running any Python file as follows:

``` markdown
source env.sh
```

## Generating character image data

We need to run through two steps to obtain the proper data format for experimenting with meta-learning algorithms.

**Step 1:** Converting the original patch-based data to character-image-based data. The converted dataset contains folders whose content is images for the corresponding Nôm characters. On top of that, there is a file called `meta_data.yaml` storing the Unicodes for these character folders.

``` markdown
python convert_data.py
```

**Step 2:** Generating splits for training, evaluation, and testing purposes:

``` markdown
python data_splitter.py
```

## Training and evaluating meta-learning algorithms

We need to define the following hyperparameters before starting training:

* `METHOD`: One of the meta-learning algorithms [protonet, maml, protomaml].
* `NUM_WAY`: Number of classes in a task.
* `NUM_SHOT`: Number of support examples per class in a task (*default* is 1).
* `INNER_LR`: Inner-loop learning rate for MAML and Proto-MAML only.
* `NUM_INNER_STEPS`: Number of inner loop updates for MAML and Proto-MAML only.
* `NUM_TRAIN_ITERATIONS`: Number of updates to train for.

Besides the training arguments, we need to define two new parameters for evaluation:

* `LOG_DIR`: Directory to load the checkpoints.
* `CHECKPOINT_STEP`: Checkpoint iteration to load for testing.

### Training

To train ProtoNet:

``` markdown
python meta_learning/run.py \
    --method protonet \
    --num_way {NUM_WAY} \
    --num_train_iterations {NUM_TRAIN_ITERATIONS}
```

To train MAML or Proto-MAML:

``` markdown
python meta_learning/run.py \
    --method {METHOD} \
    --num_way {NUM_WAY} \
    --inner_lr {INNER_LR} \
    --num_inner_steps {NUM_INNER_STEPS} \
    --num_train_iterations {NUM_TRAIN_ITERATIONS}
```

### Evaluating on the test set

To evaluate ProtoNet:

``` markdown
python meta_learning/run.py \
    --method protonet \
    --num_way {NUM_WAY} \
    --test \
    --log_dir {LOG_DIR} \
    --checkpoint_step {CHECKPOINT_STEP}
```

To evaluate MAML or Proto-MAML:

``` markdown
python meta_learning/run.py \
    --method {METHOD} \
    --num_way {NUM_WAY} \
    --inner_lr {PROTOMAML_INNER_LR} \
    --num_inner_steps {NUM_INNER_STEPS} \
    --test \
    --log_dir {LOG_DIR} \
    --checkpoint_step {CHECKPOINT_STEP}
```
## Running setup examples

We also provide example notebooks for the quick experiments

* Meta-learning with `NUM_INNER_STEPS = 1`:
    * and `NUM_WAY = 5` is provided [here](https://www.kaggle.com/code/nguynthhi/nom-character-recognition?scriptVersionId=138069640).
    * and `NUM_WAY = 10` is provided [here](https://www.kaggle.com/code/nguynthhi/nom-character-recognition?scriptVersionId=138069829).
    * and `NUM_WAY = 20` is provided [here](https://www.kaggle.com/code/nguynthhi/nom-character-recognition?scriptVersionId=138071875).
* Meta-learning with `NUM_INNER_STEPS = 5`:
    * and `NUM_WAY = 20` is provided [here](https://www.kaggle.com/code/nguynthhi/nom-character-recognition?scriptVersionId=138072064).
