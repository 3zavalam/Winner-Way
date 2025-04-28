import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, ArrowRight, PlayCircle, PauseCircle, RefreshCw } from "lucide-react";
import { VideoPlayer } from "@/components/ui/video-player";

interface TechniquePoint {
  timestamp: number;
  title: string;
  description: string;
  status: 'good' | 'improve' | 'critical';
}

interface VideoComparisonProps {
  yourVideoSrc: string;
  proVideoSrc: string;
  proPlayerName: string;
  proPlayerSpecialties: string;
  techniquePoints?: TechniquePoint[];
}

export function VideoComparison({
  yourVideoSrc,
  proVideoSrc,
  proPlayerName,
  proPlayerSpecialties,
  techniquePoints = []
}: VideoComparisonProps) {
  const [position, setPosition] = useState(50);
  const [showSlider, setShowSlider] = useState(true);
  const [selectedPoint, setSelectedPoint] = useState<TechniquePoint | null>(null);
  const [syncedPlayback, setSyncedPlayback] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef<boolean>(false);
  const yourVideoRef = useRef<HTMLVideoElement>(null);
  const proVideoRef = useRef<HTMLVideoElement>(null);

  // Handle slider dragging
  const handleMouseDown = () => {
    isDraggingRef.current = true;
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDraggingRef.current || !containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const containerWidth = rect.width;
    
    // Calculate position as percentage
    const newPosition = Math.max(0, Math.min(100, (x / containerWidth) * 100));
    setPosition(newPosition);
  };

  const handleTouchMove = (e: TouchEvent) => {
    if (!containerRef.current) return;
    
    const touch = e.touches[0];
    const rect = containerRef.current.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const containerWidth = rect.width;
    
    // Calculate position as percentage
    const newPosition = Math.max(0, Math.min(100, (x / containerWidth) * 100));
    setPosition(newPosition);
  };

  // For synchronized playback
  const togglePlayback = () => {
    if (!yourVideoRef.current || !proVideoRef.current) return;
    
    if (isPlaying) {
      yourVideoRef.current.pause();
      proVideoRef.current.pause();
    } else {
      yourVideoRef.current.play();
      proVideoRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Jump to specific technique point
  const jumpToPoint = (point: TechniquePoint) => {
    if (!yourVideoRef.current || !proVideoRef.current) return;
    
    // Set both videos to the same timestamp
    yourVideoRef.current.currentTime = point.timestamp;
    proVideoRef.current.currentTime = point.timestamp;
    
    setSelectedPoint(point);
    
    // Pause both videos at the point
    yourVideoRef.current.pause();
    proVideoRef.current.pause();
    setIsPlaying(false);
  };

  // Sync video times if enabled
  const handleTimeUpdate = (videoRef: React.RefObject<HTMLVideoElement>) => {
    if (syncedPlayback && yourVideoRef.current && proVideoRef.current) {
      if (videoRef === yourVideoRef && Math.abs(yourVideoRef.current.currentTime - proVideoRef.current.currentTime) > 0.5) {
        proVideoRef.current.currentTime = yourVideoRef.current.currentTime;
      } else if (videoRef === proVideoRef && Math.abs(proVideoRef.current.currentTime - yourVideoRef.current.currentTime) > 0.5) {
        yourVideoRef.current.currentTime = proVideoRef.current.currentTime;
      }
    }
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    document.addEventListener("mouseup", handleMouseUp);
    document.addEventListener("mousemove", handleMouseMove);
    container.addEventListener("touchmove", handleTouchMove);
    
    return () => {
      document.removeEventListener("mouseup", handleMouseUp);
      document.removeEventListener("mousemove", handleMouseMove);
      if (container) {
        container.removeEventListener("touchmove", handleTouchMove);
      }
    };
  }, []);

  // Status colors for technique points
  const getStatusColor = (status: 'good' | 'improve' | 'critical') => {
    switch (status) {
      case 'good': return 'bg-green-500';
      case 'improve': return 'bg-yellow-500';
      case 'critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row gap-6">
        {/* Left side controls */}
        <div className="md:w-1/4 space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Technique Points</CardTitle>
              <CardDescription>Click on a point to jump to that moment</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {techniquePoints.length > 0 ? (
                  techniquePoints.map((point, index) => (
                    <div 
                      key={index}
                      onClick={() => jumpToPoint(point)}
                      className={`p-2 rounded-md cursor-pointer transition-colors ${
                        selectedPoint === point 
                          ? 'bg-primary/10 border border-primary/30' 
                          : 'hover:bg-secondary/50'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{point.title}</span>
                        <Badge className={getStatusColor(point.status) + " text-white"}>
                          {point.status === 'good' ? 'Good' : point.status === 'improve' ? 'Improve' : 'Critical'}
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-500">
                        {Math.floor(point.timestamp / 60)}:{Math.floor(point.timestamp % 60).toString().padStart(2, '0')}
                      </p>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-muted-foreground">No technique points available yet.</p>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">{proPlayerName}</CardTitle>
              <CardDescription>{proPlayerSpecialties}</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm">
                Compare your technique with one of the best in the game. Use the slider to directly compare
                form and movement at any point.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Right side visualization */}
        <div className="md:w-3/4">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex justify-between items-center">
                <CardTitle className="text-lg">Side-by-Side Comparison</CardTitle>
                <div className="flex items-center gap-2">
                  <Button 
                    size="sm" 
                    variant="outline" 
                    onClick={() => setShowSlider(!showSlider)}
                  >
                    {showSlider ? "Split View" : "Slider View"}
                  </Button>
                  <Button
                    size="sm"
                    variant={syncedPlayback ? "default" : "outline"}
                    onClick={() => setSyncedPlayback(!syncedPlayback)}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    {syncedPlayback ? "Synced" : "Unsync"}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {showSlider ? (
                // Slider comparison view
                <div 
                  ref={containerRef}
                  className="relative h-80 md:h-96 rounded-lg overflow-hidden bg-black"
                >
                  {/* Left side (Your video) */}
                  <div 
                    className="absolute top-0 left-0 bottom-0 overflow-hidden"
                    style={{ width: `${position}%` }}
                  >
                    <video
                      ref={yourVideoRef}
                      src={yourVideoSrc}
                      className="w-full h-full object-cover"
                      onTimeUpdate={() => handleTimeUpdate(yourVideoRef)}
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                    />
                    <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white text-xs py-1 px-2 rounded">
                      You
                    </div>
                  </div>
                  
                  {/* Right side (Pro player video) */}
                  <div 
                    className="absolute top-0 right-0 bottom-0 overflow-hidden"
                    style={{ width: `${100 - position}%`, left: `${position}%` }}
                  >
                    <video
                      ref={proVideoRef}
                      src={proVideoSrc}
                      className="w-full h-full object-cover"
                      onTimeUpdate={() => handleTimeUpdate(proVideoRef)}
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                    />
                    <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs py-1 px-2 rounded">
                      {proPlayerName}
                    </div>
                  </div>
                  
                  {/* Slider handle */}
                  <div 
                    className="absolute top-0 bottom-0 w-1 bg-primary cursor-ew-resize z-20"
                    style={{ left: `${position}%` }}
                    onMouseDown={handleMouseDown}
                    onTouchStart={handleMouseDown}
                  >
                    <div className="absolute w-8 h-8 rounded-full bg-primary top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center text-white">
                      <ArrowLeft className="h-3 w-3" />
                      <ArrowRight className="h-3 w-3" />
                    </div>
                  </div>

                  {/* Playback control */}
                  <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20">
                    <Button 
                      variant="secondary" 
                      className="rounded-full w-12 h-12 flex items-center justify-center"
                      onClick={togglePlayback}
                    >
                      {isPlaying ? (
                        <PauseCircle className="h-6 w-6" />
                      ) : (
                        <PlayCircle className="h-6 w-6" />
                      )}
                    </Button>
                  </div>
                </div>
              ) : (
                // Split view with individual players
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="relative">
                    <VideoPlayer
                      src={yourVideoSrc}
                      className="h-80"
                    />
                    <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white text-xs py-1 px-2 rounded">
                      You
                    </div>
                  </div>
                  <div className="relative">
                    <VideoPlayer
                      src={proVideoSrc}
                      className="h-80"
                    />
                    <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white text-xs py-1 px-2 rounded">
                      {proPlayerName}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
            
          {/* Technique breakdown section */}
          {selectedPoint && (
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>{selectedPoint.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p>{selectedPoint.description}</p>
                
                <div className="mt-4">
                  <Badge className={getStatusColor(selectedPoint.status) + " text-white"}>
                    {selectedPoint.status === 'good' ? 'Good Technique' : 
                     selectedPoint.status === 'improve' ? 'Needs Improvement' : 'Critical Issue'}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}