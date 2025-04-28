import { Sidebar } from "@/components/sidebar";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Bell, HelpCircle, Play, Search, Filter, Clock, SortAsc } from "lucide-react";
import { Input } from "@/components/ui/input";
import { getVideoUrl } from "@/lib/uploadService";

export default function MyVideos() {
  const [searchTerm, setSearchTerm] = useState("");
  
  const { data: videos = [] } = useQuery({
    queryKey: ["/api/videos"],
  });

  const filteredVideos = videos.filter((video: any) => 
    video.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

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
    <div className="min-h-screen flex flex-col md:flex-row">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="px-6 py-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold text-neutral-800">My Videos</h1>
            <div className="flex items-center">
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
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="relative flex-grow">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400 h-4 w-4" />
                  <Input 
                    placeholder="Search videos..."
                    className="pl-9"
                    value={searchTerm}
                    onChange={e => setSearchTerm(e.target.value)}
                  />
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" className="flex items-center gap-2">
                    <Filter className="h-4 w-4" />
                    <span>Filter</span>
                  </Button>
                  <Button variant="outline" className="flex items-center gap-2">
                    <SortAsc className="h-4 w-4" />
                    <span>Sort</span>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {filteredVideos.length === 0 && (
            <div className="text-center py-12">
              <p className="text-neutral-400">No videos found. Try uploading one from the dashboard.</p>
              <Link href="/">
                <a>
                  <Button className="mt-4">Go to Dashboard</Button>
                </a>
              </Link>
            </div>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredVideos.map((video: any) => (
              <Card key={video.id} className="overflow-hidden">
                <Link href={`/analysis/${video.id}`}>
                  <a className="block">
                    <div className="relative aspect-video bg-neutral-200">
                      <div className="absolute inset-0 flex items-center justify-center">
                        {video.status === "processing" ? (
                          <div className="bg-black bg-opacity-60 rounded-lg px-3 py-2 text-white text-sm">
                            <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full inline-block"></div>
                            Analyzing...
                          </div>
                        ) : (
                          <Button 
                            variant="outline" 
                            size="icon" 
                            className="w-12 h-12 rounded-full bg-white/80 hover:bg-white/90"
                          >
                            <Play className="h-5 w-5 text-primary" />
                          </Button>
                        )}
                      </div>
                      <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs py-1 px-2 rounded">
                        <Clock className="h-3 w-3 inline mr-1" />
                        {video.duration ? `${Math.floor(video.duration / 60)}:${(video.duration % 60).toString().padStart(2, '0')}` : "00:00"}
                      </div>
                    </div>
                  </a>
                </Link>
                <CardContent className="p-4">
                  <h3 className="font-semibold text-lg mb-1">{video.title}</h3>
                  <div className="flex justify-between items-center mb-2">
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
                  <Progress value={video.status === "analyzed" ? 100 : 65} className="h-1" />
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
