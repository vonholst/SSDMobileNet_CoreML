//
//  utility.swift
//
//  Created by Mikael Von Holst on 2018-01-09.
//  Copyright © 2018 Mikael Von Holst. All rights reserved.
//

import Foundation

extension Double {
    func format(f: String) -> String {
        return String(format: "%\(f)f", self)
    }
}
