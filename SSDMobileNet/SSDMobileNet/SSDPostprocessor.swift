import CoreML
import UIKit

struct BoundingBox2 {
    let yMin: Double
    let xMin: Double
    let yMax: Double
    let xMax: Double
    let imgHeight = 300.0
    let imgWidth = 300.0
    
    init(yMin: Double, xMin: Double, yMax: Double, xMax: Double) {
        self.yMin = yMin
        self.xMin = xMin
        self.yMax = yMax
        self.xMax = xMax
    }
    
    init(fromAnchor: [Float32]) {
        self.yMin = Double(fromAnchor[0])
        self.xMin = Double(fromAnchor[1])
        self.yMax = Double(fromAnchor[2])
        self.xMax = Double(fromAnchor[3])
    }
    
    func toCGRect() -> CGRect {
        let height = imgHeight * (yMax - yMin)
        let width = imgWidth * (xMax - xMin)
        
        return CGRect(x: imgWidth * xMin, y: imgHeight * yMin, width: width, height: height)
    }
    func toCGRect(imgWidth:Double, imgHeight:Double, xOffset:Double, yOffset:Double) -> CGRect {
        let height = imgHeight * (yMax - yMin)
        let width = imgWidth * (xMax - xMin)
        
        return CGRect(x: imgWidth * xMin + xOffset, y: imgHeight * yMin + yOffset, width: width, height: height)
    }
}

struct AnchorEncoding {
    let ty: Double
    let tx: Double
    let th: Double
    let tw: Double
}

struct Prediction {
    let index: Int
    let score: Double
    let anchor: BoundingBox2
    let anchorEncoding: AnchorEncoding
    let detectedClass: Int
    
    var finalPrediction: BoundingBox2 {
        get {
            let yACtr = (anchor.yMin + anchor.yMax) / 2.0
            let xACtr = (anchor.xMin + anchor.xMax) / 2.0
            let ha = (anchor.yMax - anchor.yMin)
            let wa = (anchor.xMax - anchor.xMin)
            
            let ty = anchorEncoding.ty / 10.0
            let tx = anchorEncoding.tx / 10.0
            let th = anchorEncoding.th / 5.0
            let tw = anchorEncoding.tw / 5.0
            
            let w = exp(tw) * wa
            let h = exp(th) * ha
            
            let yCtr = ty * ha + yACtr
            let xCtr = tx * wa + xACtr
            
            let yMin = yCtr - h / 2.0
            let xMin = xCtr - w / 2.0
            let yMax = yCtr + h / 2.0
            let xMax = xCtr + w / 2.0
            
            return BoundingBox2(yMin: yMin, xMin: xMin, yMax: yMax, xMax: xMax)
        }
    }
}


class SSDPostProcessor {
    var numAnchors: Int
    var numClasses: Int
    var threshold: Double
    var classNames: [String]? = nil
    
    init(numAnchors: Int, numClasses: Int, threshold: Double = 0.01) {
        self.numAnchors = numAnchors
        self.numClasses = numClasses
        self.threshold = threshold
        
        if let path = Bundle.main.path(forResource: "coco_labels_list", ofType: "txt") {
            do {
                let data = try String(contentsOfFile: path, encoding: .utf8)
                classNames = data.components(separatedBy: .newlines)
                
            } catch {
                print(error)
            }
        }
        
    }
    
    func postprocess(boxPredictions: MLMultiArray, classPredictions: MLMultiArray) -> [Prediction] {
        let prunedPredictions = pruneLowScoring(boxPredictions: boxPredictions, classPredictions: classPredictions)
        
        let foo = nonMaximumSupression(predictions: prunedPredictions)
        return foo
    }
    
    private func nonMaximumSupression(predictions: [[Prediction]]) -> [Prediction] {
        var finalPredictions: [Prediction] = []
        
        for klass in 1...numClasses {
            let predictionsForClass = predictions[klass]
            
            let supressedPredictions = nonMaximumSupressionForClass(predictions: predictionsForClass, iouThreshold: 0.3, maxBoxes: 10)

            finalPredictions.append(contentsOf: supressedPredictions)
        }
        
        return finalPredictions.sorted(by: { return $0.score > $1.score })
    }
    
    private func nonMaximumSupressionForClass(_ predictions: [Prediction]) -> [Prediction] {
        return predictions
    }
    
    private func nonMaximumSupressionForClass(predictions: [Prediction],
                                              iouThreshold: Float,
                                              maxBoxes: Int) -> [Prediction] {

        // Sort the boxes based on their confidence scores, from high to low.
        let sortedPredictions = predictions.sorted { $0.score > $1.score }
        
        var selectedPredictions: [Prediction] = []
        
        // Loop through the bounding boxes, from highest score to lowest score,
        // and determine whether or not to keep each box.
        for boxA in sortedPredictions {
            if selectedPredictions.count >= maxBoxes { break }
            
            var shouldSelect = true
            
            // Does the current box overlap one of the selected boxes more than the
            // given threshold amount? Then it's too similar, so don't keep it.
            for boxB in selectedPredictions {
                if IOU(boxA.finalPrediction.toCGRect(), boxB.finalPrediction.toCGRect()) > iouThreshold {
                    shouldSelect = false
                    break
                }
            }
            
            // This bounding box did not overlap too much with any previously selected
            // bounding box, so we'll keep it.
            if shouldSelect {
                selectedPredictions.append(boxA)
            }
        }
        
        return selectedPredictions
    }

    
    private func pruneLowScoring(boxPredictions: MLMultiArray, classPredictions: MLMultiArray) -> [[Prediction]] {
        var prunedPredictions: [[Prediction]] = Array(repeating: [], count: numClasses + 1)
        
        for klass in 1...numClasses {
            for box in 0...(numAnchors-1) {
                let score = classPredictions[offset(klass, box)].doubleValue
                if score > threshold {
                    let anchor = BoundingBox2.init(fromAnchor: Anchors.ssdAnchors[box])
                    let anchorEncoding = AnchorEncoding(
                        ty: boxPredictions[offset(0, box)].doubleValue,
                        tx: boxPredictions[offset(1, box)].doubleValue,
                        th: boxPredictions[offset(2, box)].doubleValue,
                        tw: boxPredictions[offset(3, box)].doubleValue
                    )
                    
                    let prediction = Prediction(index: box, score: score, anchor: anchor, anchorEncoding: anchorEncoding, detectedClass: klass)
                    
                    prunedPredictions[klass].append(prediction)
                }
            }
        }
        
        return prunedPredictions
    }
    
    private func offset(_ i: Int, _ j: Int) -> Int {
        return i * numAnchors + j
    }
}

