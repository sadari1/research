import weka.core.jvm as jvm
from weka.core.converters import Loader
import helper as helper
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.dataset import Instances
from weka.filters import Filter

import os
#jvm.start(class_path=['/root/PycharmProjects/untitled/venv1/lib/python2.7/site-packages/weka/lib/weka.jar',
  #                    '/root/PycharmProjects/untitled/venv1/lib/python2.7/site-packages/weka/lib/python-weka-wrapper.jar'])


jvm.start()

#data_dir = "stuff/iris.arff"

#data_file = helper.get_data_dir() + os.sep + "iris.arff"
data_file = "/root/PycharmProjects/untitled/stuff/iris.arff"

helper.print_info("Loading dataset: " + data_file)
loader = Loader("weka.core.converters.ArffLoader")
data = loader.load_file(data_file)
data.class_is_last()

print(data)

classifier = Classifier(classname="weka.classifiers.trees.J48")

# randomize data
folds = 10
seed = 1
rnd = Random(seed)
rand_data = Instances.copy_instances(data)
rand_data.randomize(rnd)
if rand_data.class_attribute.is_nominal:
    rand_data.stratify(folds)

# perform cross-validation and add predictions
predicted_data = None
evaluation = Evaluation(rand_data)
for i in xrange(folds):
    train = rand_data.train_cv(folds, i)
    # the above code is used by the StratifiedRemoveFolds filter,
    # the following code is used by the Explorer/Experimenter
    # train = rand_data.train_cv(folds, i, rnd)
    test = rand_data.test_cv(folds, i)

    # build and evaluate classifier
    cls = Classifier.make_copy(classifier)
    cls.build_classifier(train)
    evaluation.test_model(cls, test)

    # add predictions
    addcls = Filter(
        classname="weka.filters.supervised.attribute.AddClassification",
        options=["-classification", "-distribution", "-error"])
    # setting the java object directory avoids issues with correct quoting in option array
    addcls.set_property("classifier", Classifier.make_copy(classifier))
    addcls.inputformat(train)
    addcls.filter(train)  # trains the classifier
    pred = addcls.filter(test)
    if predicted_data is None:
        predicted_data = Instances.template_instances(pred, 0)
    for n in xrange(pred.num_instances):
        predicted_data.add_instance(pred.get_instance(n))

print("")
print("=== Setup ===")
print("Classifier: " + classifier.to_commandline())
print("Dataset: " + data.relationname)
print("Folds: " + str(folds))
print("Seed: " + str(seed))
print("")
print(evaluation.summary("=== " + str(folds) + " -fold Cross-Validation ==="))
print("")
print(predicted_data)