import { apiRequest } from "./queryClient";

export type UploadProgressCallback = (progress: number) => void;

export const uploadVideo = async (
  file: File,
  title: string,
  onProgress?: UploadProgressCallback
): Promise<any> => {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("video", file);
    formData.append("title", title);
    
    const xhr = new XMLHttpRequest();
    
    xhr.open("POST", "/api/videos/upload", true);
    
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        const percentComplete = Math.round((event.loaded / event.total) * 100);
        onProgress(percentComplete);
      }
    };
    
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          reject(JSON.parse(xhr.responseText));
        } catch (e) {
          reject({ message: xhr.statusText });
        }
      }
    };
    
    xhr.onerror = () => {
      reject({ message: "Upload failed due to network error" });
    };
    
    xhr.send(formData);
  });
};

export const getVideoUrl = (filePath: string): string => {
  // Ensure filePath starts with a slash
  if (!filePath.startsWith("/")) {
    filePath = `/${filePath}`;
  }
  
  return filePath;
};
