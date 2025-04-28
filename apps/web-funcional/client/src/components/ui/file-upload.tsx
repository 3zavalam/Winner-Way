import React, { useState, useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { UploadCloud } from "lucide-react";

interface FileUploadProps {
  acceptedFileTypes: string;
  maxSizeMB: number;
  onFileSelected: (file: File) => void;
  onUploadProgress?: (progress: number) => void;
  isUploading?: boolean;
  uploadProgress?: number;
}

export function FileUpload({
  acceptedFileTypes,
  maxSizeMB,
  onFileSelected,
  onUploadProgress,
  isUploading = false,
  uploadProgress = 0,
}: FileUploadProps) {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    validateAndProcessFile(files[0]);
  };

  const validateAndProcessFile = (file: File) => {
    // Check file size
    if (file.size > maxSizeMB * 1024 * 1024) {
      toast({
        title: "File too large",
        description: `Maximum file size is ${maxSizeMB}MB`,
        variant: "destructive",
      });
      return;
    }

    // Check file type against accepted types
    const fileType = file.type;
    const acceptedTypes = acceptedFileTypes.split(",").map(type => type.trim());
    
    if (!acceptedTypes.some(type => fileType.includes(type.replace("*", "")))) {
      toast({
        title: "Invalid file type",
        description: `Please upload a file of type: ${acceptedFileTypes}`,
        variant: "destructive",
      });
      return;
    }

    onFileSelected(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (!files || files.length === 0) return;
    validateAndProcessFile(files[0]);
  };

  const handleClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept={acceptedFileTypes}
        className="hidden"
        disabled={isUploading}
      />
      
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer
          transition-all duration-300 bg-neutral-100 hover:bg-primary/5 hover:border-primary
          ${isDragging ? 'border-primary bg-primary/5' : 'border-neutral-300'}
          ${isUploading ? 'opacity-70 pointer-events-none' : ''}
        `}
      >
        {isUploading ? (
          <div className="text-center">
            <div className="mb-4">
              <Progress value={uploadProgress} className="w-48 h-2" />
            </div>
            <p className="text-sm text-gray-600">Uploading... {uploadProgress}%</p>
          </div>
        ) : (
          <>
            <UploadCloud className="w-12 h-12 mb-4 text-neutral-300" />
            <h3 className="font-semibold mb-2">Drag & drop your video here</h3>
            <p className="text-neutral-400 text-sm mb-4 text-center">
              or click to browse your files
            </p>
            <p className="text-xs text-neutral-400">
              Supports {acceptedFileTypes.replace(/\./g, "").toUpperCase()} up to {maxSizeMB}MB
            </p>
          </>
        )}
      </div>
    </div>
  );
}
