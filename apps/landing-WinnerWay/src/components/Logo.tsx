
import React from 'react';

const Logo: React.FC = () => {
  return (
    <div className="flex items-center gap-2">
      <img
        src="/lovable-uploads/4bdb015c-2be1-4861-9cd3-9704d24943ea.png"
        alt="Winner Way Logo"
        className="h-10 w-auto"
      />
      <span className="font-bold text-xl text-winner-green">Winner Way</span>
    </div>
  );
};

export default Logo;
