import React from 'react';

const Hero: React.FC = () => {
  const scrollToUpload = () => {
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
      uploadSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="section pt-10 md:pt-16">
      <div className="winner-container">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          <div className="animate-fade-in">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-winner-green mb-6">
              Improve Your Tennis Technique with AI
            </h1>
            <p className="text-lg md:text-xl text-winner-green/80 mb-4">
              Try the AI Demo Below – No Sign Up Needed
            </p>
            <p className="text-xl md:text-2xl text-winner-green/80 mb-8">
              Upload your video, compare it to a pro, and get feedback in seconds.
            </p>
            <div className="flex flex-wrap gap-4">
              <button onClick={scrollToUpload} className="btn-primary">Get Started</button>
              <a href="#how-it-works" className="btn-secondary">
                Learn More
              </a>
            </div>
            {/* Flecha animada justo aquí, como lo tenías antes */}
            <div
              onClick={scrollToUpload}
              className="text-center mt-10 text-3xl text-winner-green cursor-pointer animate-bounce"
            >
              ↓
            </div>
          </div>
          <div className="flex justify-center md:justify-end">
            <img
              src="/lovable-uploads/935ad86f-b91b-4e67-a0c3-b3a93b7aff72.png"
              alt="Friendly Tennis Coach Mascot"
              className="w-3/4 md:w-full max-w-md animate-bounce-subtle"
            />
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;