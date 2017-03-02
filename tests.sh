caffe test -model test.prototxt -weights resnet50_cvgj_iter_$1.caffemodel -gpu 0 -iterations 1000 2> test-1000-`date +%F`.log
