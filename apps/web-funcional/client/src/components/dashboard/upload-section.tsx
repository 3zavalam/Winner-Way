import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { FileUpload } from "@/components/ui/file-upload";
import { useToast } from "@/hooks/use-toast";
import { uploadVideo } from "@/lib/uploadService";
import { Video, Play, Clock } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "wouter";
import { getVideoUrl } from "@/lib/uploadService";

export function UploadSection() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const { data: videos = [] } = useQuery({
    queryKey: ["/api/videos"],
  });

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    // Set title from filename (without extension)
    const fileName = file.name.replace(/\.[^/.]+$/, "");
    setTitle(fileName);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please select a video file to upload",
        variant: "destructive",
      });
      return;
    }

    if (!title.trim()) {
      toast({
        title: "Title required",
        description: "Please provide a title for your video",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);
    
    try {
      await uploadVideo(selectedFile, title, (progress) => {
        setUploadProgress(progress);
      });
      
      toast({
        title: "Upload successful",
        description: "Your video is now being processed for analysis",
      });
      
      // Reset the form
      setSelectedFile(null);
      setTitle("");
      setUploadProgress(0);
      
      // Invalidate videos query to refresh the list
      queryClient.invalidateQueries({ queryKey: ["/api/videos"] });
      
    } catch (error: any) {
      toast({
        title: "Upload failed",
        description: error.message || "An error occurred during upload",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const formatTimeSince = (dateString: string) => {
    const uploadedDate = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - uploadedDate.getTime();
    
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffDays > 0) {
      return `${diffDays} ${diffDays === 1 ? 'day' : 'days'} ago`;
    } else if (diffHours > 0) {
      return `${diffHours} ${diffHours === 1 ? 'hour' : 'hours'} ago`;
    } else {
      return `${diffMins} ${diffMins === 1 ? 'minute' : 'minutes'} ago`;
    }
  };

  return (
    <section className="mb-8">
      <Card>
        <CardContent className="p-6">
          <h2 className="text-xl font-semibold mb-4">Upload Your Tennis Video</h2>
          
          <FileUpload
            acceptedFileTypes="video/mp4,video/quicktime,video/x-msvideo"
            maxSizeMB={500}
            onFileSelected={handleFileSelected}
            isUploading={isUploading}
            uploadProgress={uploadProgress}
          />
          
          {selectedFile && !isUploading && (
            <div className="mt-4 flex justify-between items-center">
              <div className="flex items-center">
                <Video className="h-5 w-5 mr-2 text-primary" />
                <span className="text-sm font-medium mr-2">{selectedFile.name}</span>
                <span className="text-xs text-neutral-400">
                  ({Math.round(selectedFile.size / 1024 / 1024 * 10) / 10} MB)
                </span>
              </div>
              <button
                onClick={handleUpload}
                className="px-4 py-2 rounded-lg bg-primary text-white text-sm hover:bg-primary/90 transition-colors"
              >
                Upload
              </button>
            </div>
          )}
          
          {/* Recently Uploaded Videos */}
          {videos.length > 0 && (
            <div className="mt-8">
              <h3 className="font-semibold mb-4">Recently Uploaded</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {videos.map((video: any) => (
                  <div key={video.id} className="bg-neutral-100 rounded-lg overflow-hidden">
                    <Link href={`/analysis/${video.id}`}>
                      <a className="block">
                        <div className="relative h-40 bg-neutral-200">
                          <div className="absolute inset-0 flex items-center justify-center">
                            {video.status === "processing" ? (
                              <div className="bg-black bg-opacity-60 rounded-lg px-3 py-2 text-white text-sm">
                                <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full inline-block"></div>
                                Analyzing...
                              </div>
                            ) : (
                              <button className="bg-white bg-opacity-80 rounded-full w-12 h-12 flex items-center justify-center">
                                <Play className="h-5 w-5 text-primary" />
                              </button>
                            )}
                          </div>
                          <div className="absolute bottom-0 right-0 bg-black bg-opacity-70 text-white text-xs py-1 px-2 rounded-tl-md">
                            <Clock className="h-3 w-3 inline mr-1" />
                            {video.duration ? `${Math.floor(video.duration / 60)}:${(video.duration % 60).toString().padStart(2, '0')}` : "00:00"}
                          </div>
                        </div>
                      </a>
                    </Link>
                    <div className="p-3">
                      <h4 className="font-semibold mb-1">{video.title}</h4>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-neutral-400">
                          Uploaded {formatTimeSince(video.uploadedAt)}
                        </span>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          video.status === "analyzed" 
                            ? "bg-green-100 text-green-600" 
                            : "bg-yellow-100 text-yellow-600"
                        }`}>
                          {video.status === "analyzed" ? "Analyzed" : "Processing"}
                        </span>
                      </div>
                      <div className="mt-2">
                        <Progress value={video.status === "analyzed" ? 100 : 65} className="h-1" />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </section>
  );
}
