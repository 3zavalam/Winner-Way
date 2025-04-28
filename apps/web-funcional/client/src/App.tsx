import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import MyVideos from "@/pages/my-videos";
import ProLibrary from "@/pages/pro-library";
import VideoAnalysis from "@/pages/video-analysis";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/my-videos" component={MyVideos} />
      <Route path="/pro-library" component={ProLibrary} />
      <Route path="/analysis/:id" component={VideoAnalysis} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router />
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
