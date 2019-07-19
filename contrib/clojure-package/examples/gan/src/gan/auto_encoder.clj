(ns gan.auto-encoder
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.ndarray-api :as ndarray-api]
            [org.apache.clojure-mxnet.image :as image]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [gan.viz :as viz]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.optimizer :as optimizer])
  (:import (javax.imageio ImageIO)
           (org.apache.mxnet Image NDArray)
           (java.io File)))

(def data-dir "data/")
(def batch-size 100)

(when-not (.exists (io/file (str data-dir "train-images-idx3-ubyte")))
  (sh "../../scripts/get_mnist_data.sh"))


;;; Load the MNIST datasets
;;; note that the label is the same as the image
(defonce train-data (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                                       :label (str data-dir "train-labels-idx1-ubyte")
                                       :label-name "softmax_label"
                                       :input-shape [784]
                                       :batch-size batch-size
                                       :shuffle true
                                       :flat true
                                       :silent false
                                       :seed 10}))

(defonce test-data (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                                      :label (str data-dir "t10k-labels-idx1-ubyte")
                                      :input-shape [784]
                                      :batch-size batch-size
                                      :flat true
                                      :silent false}))

(def output (sym/variable "input_"))

(comment
  (def x (mx-io/next test-data))

  (mx-io/batch-label x)


  )

(defn get-symbol []
  (as-> (sym/variable "input") data
    ;; decode
    (sym/fully-connected "decode1" {:data data :num-hidden 50})
    (sym/activation "sigmoid3" {:data data :act-type "sigmoid"})

    ;; decode
    (sym/fully-connected "decode2" {:data data :num-hidden 100})
    (sym/activation "sigmoid4" {:data data :act-type "sigmoid"})

    ;;output
    (sym/fully-connected "result" {:data data :num-hidden 784})
    (sym/activation "sigmoid5" {:data data :act-type "sigmoid"})

    (sym/linear-regression-output {:data data :label output})

    ))

(def data-desc (first (mx-io/provide-data-desc train-data)))
(def label-desc (first (mx-io/provide-label-desc train-data)))

(def model (-> (m/module (get-symbol) {:data-names ["input"] :label-names ["input_"]})
               (m/bind {:data-shapes [(assoc label-desc :name "input")]
                        :label-shapes [(assoc data-desc :name "input_")]})
               (m/init-params {:initializer  (initializer/uniform 1)})
               (m/init-optimizer {:optimizer (optimizer/adam {:learning-rage 0.001})})))

(def my-metric (eval-metric/mse))

(defn train [num-epochs]
  (doseq [epoch-num (range 0 num-epochs)]
    (println "starting epoch " epoch-num)
    (mx-io/do-batches
     train-data
     (fn [batch]
       (-> model
           (m/forward {:data (mx-io/batch-label batch) :label (mx-io/batch-data batch)})
           (m/update-metric my-metric (mx-io/batch-data batch))
           (m/backward)
           (m/update))))
    (println "result for epoch " epoch-num " is " (eval-metric/get-and-reset my-metric))))

(comment

  (mx-io/provide-data train-data)
  (mx-io/provide-label train-data)
  (mx-io/reset train-data)
  (def my-batch (mx-io/next train-data))
  (def train-labels (mx-io/batch-label my-batch))
  (def images (mx-io/batch-data my-batch))
  (ndarray/shape (ndarray/reshape (first images) [100 1 28 28]))
  (viz/im-sav {:title "originals" :output-path "results/" :x (ndarray/reshape (first images) [100 1 28 28])})


  (ndarray/shape (first train-labels))
  (train 3)


  (def my-test-batch (mx-io/next test-data))
  (def test-images (mx-io/batch-data my-test-batch))
  (def test-labels (mx-io/batch-label my-test-batch))
  (def preds (m/predict-batch model {:data test-labels} ))
  (viz/im-sav {:title "preds" :output-path "results/" :x (ndarray/reshape (first preds) [100 1 28 28])})

  (def x (ndarray/array (into [] (repeat 100 2)) [100]))
  (def new-preds (m/predict-batch model {:data [x]} ))
  (viz/im-sav {:title "preds" :output-path "results/" :x (ndarray/reshape (first new-preds) [100 1 28 28])})


  (ndarray/slice (first test-labels) 0 1) 
  (ndarray/slice (first preds) 0 1)

  (first images)
  (ndarray/div (first train-labels) 255.0)
  (ndarray/as-type (first train-labels) dtype/UINT8)
  (-> (ndarray/to-scalar (ndarray/slice (first test-labels) 1 2))
      (/ 255.0))

    (-> (ndarray/to-scalar (ndarray/slice (first train-labels) 0 1)))

  (/ 231.0 255)


  (sym/list-arguments (m/symbol my-mod))
  (def data-desc (first (mx-io/provide-data-desc train-data)))






  (let [mod (m/module (get-sy) {:contexts devs})]
    ;;; note only one function for training
    (m/fit mod {:train-data train-data :eval-data test-data :num-epoch num-epoch})

    ;;high level predict (just a dummy call but it returns a vector of results
    (m/predict mod {:eval-data test-data})

    ;;;high level score (returs the eval values)
    (let [score (m/score mod {:eval-data test-data :eval-metric (eval-metric/accuracy)})]
      (println "High level predict score is " score)))

  


  
  )

;;; Autoencoder network
;;; input -> encode -> middle -> decode -> output
