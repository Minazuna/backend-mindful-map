const leoProfanity = require('leo-profanity');

function getFilter() {
  leoProfanity.loadDictionary(); 
  leoProfanity.add(['damn', 'hell', 'crap', 'idiot', 'stupid']);
  
  return {
    check: (text) => {
      // Check if any word in the text is profane
      const words = text.split(' ');
      for (const word of words) {
        if (leoProfanity.check(word)) {
          return true;
        }
      }
      return false;
    },
    
    // Keep the clean function for other uses if needed
    clean: (text) => {
      return text.split(' ').map(word => {
        if (leoProfanity.check(word)) {
          return word[0] + '*'.repeat(word.length - 1); // Keep first letter, mask the rest
        }
        return word;
      }).join(' ');
    }
  };
}

module.exports = { getFilter };