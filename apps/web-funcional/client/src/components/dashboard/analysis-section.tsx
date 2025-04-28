import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { VideoPlayer } from "@/components/ui/video-player";
import { Button } from "@/components/ui/button";
import { ComparisonSlider } from "./comparison-slider";
import { Progress } from "@/components/ui/progress";
import { useQuery } from "@tanstack/react-query";
import { Camera, StepBack, StepForward, Check, AlertCircle, Bot } from "lucide-react";
import { getVideoUrl } from "@/lib/uploadService";

export function AnalysisSection() {
  const [selectedThumbnailIndex, setSelectedThumbnailIndex] = useState(1);
  const [selectedProPlayer, setSelectedProPlayer] = useState("Rafael Nadal");

  // Get the latest analyzed video
  const { data: videos = [] } = useQuery({
    queryKey: ["/api/videos"],
  });

  // Find the first analyzed video
  const latestAnalyzedVideo = videos.find((video: any) => video.status === "analyzed");

  // Get analysis data for the video
  const { data: analysis } = useQuery({
    queryKey: [
      `/api/videos/${latestAnalyzedVideo?.id}/analysis`,
    ],
    enabled: !!latestAnalyzedVideo,
  });

  // Get pro players data
  const { data: proPlayers = [] } = useQuery({
    queryKey: ["/api/pro-players"],
  });

  if (!latestAnalyzedVideo || !analysis) {
    return (
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Latest Analysis</h2>
        <Card>
          <CardContent className="p-12 text-center">
            <div className="flex flex-col items-center justify-center">
              <Bot className="h-12 w-12 text-neutral-300 mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Analysis Available</h3>
              <p className="text-neutral-400 max-w-md">
                Upload a tennis video to receive AI-powered feedback and technique analysis.
              </p>
            </div>
          </CardContent>
        </Card>
      </section>
    );
  }

  // Mock frame thumbnails (in a real app, these would come from the backend)
  const frameThumbnails = Array(5).fill(null).map((_, i) => ({
    id: i,
    url: getVideoUrl(latestAnalyzedVideo.filePath),
    isSelected: i === selectedThumbnailIndex
  }));

  return (
    <section className="mb-8">
      <h2 className="text-xl font-semibold mb-4">Latest Analysis</h2>
      
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row">
            <div className="w-full lg:w-7/12 mb-6 lg:mb-0 lg:pr-6">
              <VideoPlayer 
                src={getVideoUrl(latestAnalyzedVideo.filePath)}
                className="rounded-lg mb-4 aspect-video"
              />
              
              {/* Frame navigation */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold">Frame Analysis</h3>
                  <div className="flex">
                    <Button 
                      variant="outline" 
                      size="icon" 
                      className="mr-2" 
                      onClick={() => setSelectedThumbnailIndex(Math.max(0, selectedThumbnailIndex - 1))}
                    >
                      <StepBack className="h-4 w-4" />
                    </Button>
                    <Button 
                      variant="outline" 
                      size="icon" 
                      className="mr-2" 
                      onClick={() => setSelectedThumbnailIndex(Math.min(frameThumbnails.length - 1, selectedThumbnailIndex + 1))}
                    >
                      <StepForward className="h-4 w-4" />
                    </Button>
                    <Button size="icon" className="bg-primary text-white hover:bg-primary/90">
                      <Camera className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                {/* Frame thumbnails */}
                <div className="flex overflow-x-auto py-2 -mx-2">
                  {frameThumbnails.map((frame) => (
                    <div key={frame.id} className="flex-shrink-0 px-2">
                      <div 
                        className={`w-20 h-20 rounded-lg overflow-hidden cursor-pointer border-2 ${
                          frame.isSelected ? 'border-primary' : 'border-transparent hover:border-primary'
                        }`}
                        onClick={() => setSelectedThumbnailIndex(frame.id)}
                      >
                        <div className="w-full h-full bg-neutral-200" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="w-full lg:w-5/12 lg:pl-6">
              <div className="mb-6">
                <h3 className="font-semibold mb-4">AI Analysis</h3>
                <div className="bg-neutral-100 rounded-lg p-4">
                  <div className="flex items-start mb-4">
                    <div className="bg-secondary rounded-full p-2 mr-3 text-white">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div>
                      <h4 className="font-semibold mb-1">Forehand Technique</h4>
                      <p className="text-sm text-neutral-500">
                        {analysis.feedback}
                      </p>
                    </div>
                  </div>
                  
                  <div className="mb-4 border-t border-neutral-200 pt-4">
                    <h4 className="font-semibold mb-2 text-sm">Key Points:</h4>
                    <ul className="text-sm text-neutral-500 space-y-2">
                      {analysis.keyPoints.map((point: string, index: number) => (
                        <li key={index} className="flex items-start">
                          {point.includes("could") || point.includes("Consider") ? (
                            <AlertCircle className="text-amber-500 h-4 w-4 mt-1 mr-2" />
                          ) : (
                            <Check className="text-green-500 h-4 w-4 mt-1 mr-2" />
                          )}
                          <span>{point}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="border-t border-neutral-200 pt-4">
                    <h4 className="font-semibold mb-2 text-sm">Drills to Improve:</h4>
                    <div className="text-sm text-neutral-500 space-y-2">
                      {analysis.drills.map((drill: string, index: number) => (
                        <p key={index}>{index + 1}. {drill}</p>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-semibold">Compare with Pro</h3>
                  <div className="relative">
                    <select 
                      className="bg-neutral-100 border-0 rounded-lg py-2 pl-3 pr-8 appearance-none focus:outline-none focus:ring-2 focus:ring-primary text-sm"
                      value={selectedProPlayer}
                      onChange={e => setSelectedProPlayer(e.target.value)}
                    >
                      {proPlayers.map((player: any) => (
                        <option key={player.id} value={player.name}>{player.name}</option>
                      ))}
                    </select>
                    <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none">
                      <svg className="h-4 w-4 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>
                </div>
                
                <ComparisonSlider
                  yourImgSrc={getVideoUrl(latestAnalyzedVideo.filePath)}
                  proImgSrc="https://images.unsplash.com/photo-1567226051112-173ae646173c?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80"
                  proName={selectedProPlayer}
                />
                
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div className="bg-neutral-100 rounded-lg p-3">
                    <h4 className="font-semibold text-sm mb-1">Angle Comparison</h4>
                    <div className="flex items-center">
                      <div className="w-full bg-neutral-200 rounded-full h-2 mr-2">
                        <div className="bg-primary h-2 rounded-full" style={{ width: `${analysis.techniqueScores.angle}%` }}></div>
                      </div>
                      <span className="text-sm text-neutral-500">{analysis.techniqueScores.angle}%</span>
                    </div>
                  </div>
                  <div className="bg-neutral-100 rounded-lg p-3">
                    <h4 className="font-semibold text-sm mb-1">Power</h4>
                    <div className="flex items-center">
                      <div className="w-full bg-neutral-200 rounded-full h-2 mr-2">
                        <div className="bg-primary h-2 rounded-full" style={{ width: `${analysis.techniqueScores.power}%` }}></div>
                      </div>
                      <span className="text-sm text-neutral-500">{analysis.techniqueScores.power}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
