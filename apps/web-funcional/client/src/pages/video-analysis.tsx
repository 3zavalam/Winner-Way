import { useState } from "react";
import { useRoute } from "wouter";
import { Sidebar } from "@/components/sidebar";
import { VideoPlayer } from "@/components/ui/video-player";
import { VideoComparison } from "@/components/video-analysis/video-comparison";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Bell, HelpCircle, ArrowLeft, Camera, StepBack, StepForward, Download, Share2, Check, AlertCircle, Bot } from "lucide-react";
import { Link } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { getVideoUrl } from "@/lib/uploadService";

export default function VideoAnalysis() {
  const [, params] = useRoute("/analysis/:id");
  const videoId = params?.id ? parseInt(params.id) : 0;
  
  const [selectedProPlayer, setSelectedProPlayer] = useState("Rafael Nadal");
  const [selectedThumbnailIndex, setSelectedThumbnailIndex] = useState(1);

  // Get video data
  const { data: video, isLoading: isLoadingVideo } = useQuery({
    queryKey: [`/api/videos/${videoId}`],
    enabled: !!videoId,
  });

  // Get analysis data
  const { data: analysis, isLoading: isLoadingAnalysis } = useQuery({
    queryKey: [`/api/videos/${videoId}/analysis`],
    enabled: !!videoId && video?.status === "analyzed",
  });

  // Get pro players
  const { data: proPlayers = [] } = useQuery({
    queryKey: ["/api/pro-players"],
  });

  // Mock frame thumbnails (in a real app, these would come from the backend)
  const frameThumbnails = Array(5).fill(null).map((_, i) => ({
    id: i,
    url: video ? getVideoUrl(video.filePath) : "",
    isSelected: i === selectedThumbnailIndex
  }));

  if (isLoadingVideo) {
    return (
      <div className="min-h-screen flex flex-col md:flex-row">
        <Sidebar />
        <main className="flex-1 p-6 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full inline-block mb-4"></div>
            <p>Loading video data...</p>
          </div>
        </main>
      </div>
    );
  }

  if (!video) {
    return (
      <div className="min-h-screen flex flex-col md:flex-row">
        <Sidebar />
        <main className="flex-1 p-6 flex items-center justify-center">
          <Card>
            <CardContent className="p-12 text-center">
              <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
              <h2 className="text-xl font-semibold mb-2">Video Not Found</h2>
              <p className="text-neutral-500 mb-6">The video you're looking for doesn't exist or has been removed.</p>
              <Link href="/">
                <a>
                  <Button>Return to Dashboard</Button>
                </a>
              </Link>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col md:flex-row">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="px-6 py-4 flex justify-between items-center">
            <div className="flex items-center">
              <Link href="/my-videos">
                <a>
                  <Button variant="ghost" size="icon" className="mr-2">
                    <ArrowLeft className="h-5 w-5" />
                  </Button>
                </a>
              </Link>
              <h1 className="text-2xl font-bold text-neutral-800">{video.title}</h1>
            </div>
            <div className="flex items-center">
              <Button variant="ghost" size="icon" className="text-neutral-400 mr-2">
                <Download className="h-5 w-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-neutral-400 mr-2">
                <Share2 className="h-5 w-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-neutral-400 mr-4">
                <Bell className="h-5 w-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-neutral-400">
                <HelpCircle className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </header>

        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card className="mb-6">
                <CardContent className="p-6">
                  <VideoPlayer 
                    src={getVideoUrl(video.filePath)}
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
                </CardContent>
              </Card>
              
              {video.status === "analyzed" && analysis && (
                <Card>
                  <CardContent className="p-6">
                    <h3 className="font-semibold mb-4">Compare with Pro</h3>
                    
                    <div className="flex justify-between items-center mb-4">
                      <p className="text-sm text-neutral-500">Compare your technique with the pros to see the difference</p>
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
                    
                    {/* Enhanced interactive video comparison */}
                    <VideoComparison
                      yourVideoSrc={getVideoUrl(video.filePath)}
                      proVideoSrc={getVideoUrl(video.filePath)}
                      proPlayerName={selectedProPlayer}
                      proPlayerSpecialties={
                        proPlayers.find((player: any) => player.name === selectedProPlayer)?.specialties || 
                        "Professional Tennis Player"
                      }
                      techniquePoints={[
                        // Sample technique points - in a real app, these would come from analysis data
                        {
                          timestamp: 2,
                          title: "Forehand Grip",
                          description: "Your grip is properly positioned with the eastern forehand grip, allowing for good control and spin generation.",
                          status: 'good'
                        },
                        {
                          timestamp: 5,
                          title: "Backswing",
                          description: "Your backswing is slightly too high, reducing power potential. Try to keep the racket head below shoulder height on the backswing.",
                          status: 'improve'
                        },
                        {
                          timestamp: 8,
                          title: "Weight Transfer",
                          description: "You're not fully transferring your weight forward during the shot, losing potential power. Make sure to step into the ball.",
                          status: 'critical'
                        },
                        {
                          timestamp: 12,
                          title: "Follow Through",
                          description: "Good follow-through motion, carrying the racket across your body for a complete stroke.",
                          status: 'good'
                        }
                      ]}
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
                  </CardContent>
                </Card>
              )}
            </div>
            
            <div>
              {video.status === "processing" ? (
                <Card>
                  <CardContent className="p-6">
                    <div className="text-center py-8">
                      <div className="animate-spin h-12 w-12 border-4 border-primary border-t-transparent rounded-full inline-block mb-4"></div>
                      <h3 className="text-lg font-semibold mb-2">Analyzing Your Video</h3>
                      <p className="text-neutral-500 mb-4">
                        Our AI is analyzing your tennis technique. This may take a few minutes.
                      </p>
                      <Progress value={65} className="mb-4" />
                      <p className="text-sm text-neutral-400">
                        Please wait while we process your video and provide detailed feedback.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="p-6">
                    <h3 className="font-semibold mb-4">AI Analysis</h3>
                    
                    <div className="bg-neutral-100 rounded-lg p-4">
                      <div className="flex items-start mb-4">
                        <div className="bg-secondary rounded-full p-2 mr-3 text-white">
                          <Bot className="h-4 w-4" />
                        </div>
                        <div>
                          <h4 className="font-semibold mb-1">Forehand Technique</h4>
                          <p className="text-sm text-neutral-500">
                            {analysis?.feedback || "Analysis not available yet."}
                          </p>
                        </div>
                      </div>
                      
                      {analysis && (
                        <>
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
                        </>
                      )}
                    </div>
                    
                    {analysis && (
                      <div className="mt-6">
                        <h3 className="font-semibold mb-4">Performance Metrics</h3>
                        
                        <div className="space-y-4">
                          {Object.entries(analysis.techniqueScores).map(([key, value]: [string, any]) => (
                            <div key={key}>
                              <div className="flex justify-between items-center mb-1">
                                <span className="text-sm capitalize">{key}</span>
                                <span className="text-sm font-semibold">{value}%</span>
                              </div>
                              <Progress value={value} />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
