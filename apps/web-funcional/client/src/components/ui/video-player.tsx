import React, { useState, useRef, useEffect } from "react";
import { Play, Pause, Maximize, SkipForward, SkipBack } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

interface VideoPlayerProps {
  src: string;
  poster?: string;
  onTimeUpdate?: (currentTime: number, duration: number) => void;
  onEnded?: () => void;
  className?: string;
}

export function VideoPlayer({
  src,
  poster,
  onTimeUpdate,
  onEnded,
  className,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [loaded, setLoaded] = useState(false);

  const formatTime = (seconds: number): string => {
    if (isNaN(seconds)) return "00:00";
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    return `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const current = videoRef.current.currentTime;
      setCurrentTime(current);
      
      if (onTimeUpdate) {
        onTimeUpdate(current, duration);
      }
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      setLoaded(true);
    }
  };

  const handleSliderChange = (value: number[]) => {
    if (videoRef.current) {
      const newTime = (value[0] / 100) * duration;
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    if (onEnded) {
      onEnded();
    }
  };

  const handleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen();
      }
    }
  };

  const skipForward = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.min(videoRef.current.duration, videoRef.current.currentTime + 5);
    }
  };

  const skipBackward = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime - 5);
    }
  };

  return (
    <div className={`relative rounded-lg overflow-hidden bg-black ${className}`}>
      <video
        ref={videoRef}
        src={src}
        poster={poster}
        className="w-full h-full object-contain"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
      />
      
      <div className="absolute inset-0 flex items-center justify-center">
        {!isPlaying && (
          <Button 
            variant="outline" 
            size="icon" 
            className="w-16 h-16 rounded-full bg-white/80 hover:bg-white/90" 
            onClick={togglePlay}
          >
            <Play className="h-8 w-8 text-primary" />
          </Button>
        )}
      </div>
      
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
        <Slider 
          value={[loaded ? (currentTime / duration) * 100 : 0]} 
          onValueChange={handleSliderChange}
          className="h-1 mb-2"
        />
        
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Button size="icon" variant="ghost" className="text-white h-8 w-8" onClick={togglePlay}>
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button size="icon" variant="ghost" className="text-white h-8 w-8" onClick={skipBackward}>
              <SkipBack className="h-4 w-4" />
            </Button>
            <Button size="icon" variant="ghost" className="text-white h-8 w-8" onClick={skipForward}>
              <SkipForward className="h-4 w-4" />
            </Button>
            <span className="text-white text-xs">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>
          
          <Button size="icon" variant="ghost" className="text-white h-8 w-8" onClick={handleFullscreen}>
            <Maximize className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
