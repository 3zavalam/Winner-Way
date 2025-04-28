import { useState, useRef, useEffect } from "react";

interface ComparisonSliderProps {
  yourImgSrc: string;
  proImgSrc: string;
  proName: string;
}

export function ComparisonSlider({ yourImgSrc, proImgSrc, proName }: ComparisonSliderProps) {
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef<boolean>(false);

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

  return (
    <div 
      ref={containerRef}
      className="relative h-48 md:h-64 rounded-lg overflow-hidden"
    >
      {/* Left side (Your image) */}
      <div 
        className="absolute top-0 left-0 bottom-0 overflow-hidden"
        style={{ width: `${position}%` }}
      >
        <div className="absolute top-0 left-0 right-0 bottom-0 bg-neutral-200"></div>
        <img 
          src={yourImgSrc} 
          alt="Your technique" 
          className="w-full h-full object-cover"
          onError={(e) => {
            // If image fails to load, show a neutral background
            e.currentTarget.style.display = "none";
          }}
        />
        <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white text-xs py-1 px-2 rounded">
          You
        </div>
      </div>
      
      {/* Right side (Pro player image) */}
      <div 
        className="absolute top-0 right-0 bottom-0 overflow-hidden"
        style={{ width: `${100 - position}%`, left: `${position}%` }}
      >
        <div className="absolute top-0 left-0 right-0 bottom-0 bg-neutral-300"></div>
        <img 
          src={proImgSrc} 
          alt={`${proName}'s technique`}
          className="w-full h-full object-cover"
          onError={(e) => {
            // If image fails to load, show a neutral background
            e.currentTarget.style.display = "none";
          }}
        />
        <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs py-1 px-2 rounded">
          {proName}
        </div>
      </div>
      
      {/* Slider handle */}
      <div 
        className="absolute top-0 bottom-0 w-1 bg-primary cursor-ew-resize"
        style={{ left: `${position}%` }}
        onMouseDown={handleMouseDown}
        onTouchStart={handleMouseDown}
      >
        <div className="absolute w-6 h-6 rounded-full bg-primary top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <div className="absolute inset-0 rounded-full shadow-lg opacity-30 bg-primary animate-pulse"></div>
        </div>
      </div>
    </div>
  );
}
