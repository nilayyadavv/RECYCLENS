//
//  ContentView.swift
//  RecycLense
//
//  Created by Ayush Sadekar on 3/29/25.
//

import SwiftUI
import PhotosUI
import UIKit

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var showingCamera = false
    @State private var isUploading = false // State for showing the loading screen
    @State private var classificationResult: String? // State to hold the classification result
    @State private var showResultScreen = false // State to show the result screen
    
    var body: some View {
        if showResultScreen {
            ResultView(classificationResult: classificationResult, onGoBack: {
                showResultScreen = false
                classificationResult = nil
                selectedImage = nil
            })
        } else {
            VStack {
                if let selectedImage = selectedImage {
                    Image(uiImage: selectedImage)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                        .cornerRadius(25)
                } else {
                    Text("No image selected")
                        .foregroundStyle(.gray)
                        .padding()
                }

                Button(action: {
                    showingCamera = true
                }) {
                    Text("Take Photo")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color("Secondary Color"))
                        .foregroundColor(.white)
                        .cornerRadius(25)
                }
                .sheet(isPresented: $showingCamera) {
                    CameraView(image: $selectedImage)
                }

                PhotosPicker(selection: $selectedItem, matching: .images, photoLibrary: .shared()) {
                    Text("Select Photo")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color("Secondary Color"))
                        .foregroundColor(.white)
                        .cornerRadius(25)
                    
                }
                

                Button(action: {
                    if let image = selectedImage {
                        uploadImage(image: image) // Call the upload function
                    }
                }) {
                    Text("Upload Photo")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(25)
                }
                
                Image("RL_Logo").resizable().scaledToFit().frame(height:200)
            }
            .onChange(of: selectedItem) { newItem in
                if let newItem = newItem {
                    Task {
                        if let data = try? await newItem.loadTransferable(type: Data.self),
                           let image = UIImage(data: data) {
                            selectedImage = image
                        }
                    }
                }
            }
            .overlay(
                Group {
                    if isUploading {
                        LoadingView() // Show loading screen while uploading
                    }
                }
            )
            .padding()
        }
    }

    func uploadImage(image: UIImage) {
        guard let url = URL(string: "http://10.142.42.241:5050/") else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        guard let imageData = image.jpegData(compressionQuality: 0.8) else { return } // Convert UIImage to JPEG data
        
        var body = Data()
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        isUploading = true // Show loading screen
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isUploading = false // Hide loading screen
                
                if let error = error {
                    print("Error uploading image:", error.localizedDescription)
                    return
                }
                
                if let data = data,
                   let resultJSON = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                   let classification = resultJSON["classification"] as? String {
                    classificationResult = classification // Store the classification result
                    showResultScreen = true // Show result screen
                } else {
                    print("Failed to parse response.")
                }
            }
        }
        
        task.resume() // Start the upload task
    }
}

struct LoadingView: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.5).edgesIgnoringSafeArea(.all) // Semi-transparent background
            
            VStack(spacing: 20) {
                ProgressView() // Loading spinner
                    .scaleEffect(2.0) // Make spinner larger
                
                Text("Processing...")
                    .font(.headline)
                    .foregroundColor(.white)
            }
        }
    }
}

struct ResultView: View {
    let classificationResult: String?
    let onGoBack: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Text("Classification Result")
                .font(.title)

            if let result = classificationResult {
                Text(result)
                    .font(.headline)
                    .padding()
                    .background(Color.green.opacity(0.3))
                    .cornerRadius(10)
            } else {
                Text("No result available.")
                    .font(.headline)
                    .foregroundColor(.gray)
            }

            Button(action: onGoBack) { // Button to go back to ContentView
                Text("Go Back")
                    .font(.headline)
                    .padding()
                    .frame(maxWidth: 200)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(25)
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

