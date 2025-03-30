//
//  CameraView.swift
//  RecycLense
//
//  Created by Ayush Sadekar on 3/29/25.
//

import Foundation
import SwiftUI
import UIKit

struct CameraView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController() // create Camera Picker
        picker.delegate = context.coordinator // set the coordinator as delegate
        picker.sourceType = .camera // set source to camera
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // No updates needed
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: CameraView
        
        init(_ parent: CameraView){
            self.parent = parent
        }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]){
            if let image = info[.originalImage] as? UIImage{
                parent.image = image //
            }
            parent.presentationMode.wrappedValue.dismiss() // dismiss the picker
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss() // Dismiss on cancel
        }
    }
}
