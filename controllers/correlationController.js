const moment = require('moment');
const MoodLog = require('../models/MoodLog');
const Correlation = require('../models/Correlation');
const CorrelationValue = require('../models/CorrelationValue');

const recommendationsData = {
  moods: {
    relaxed: [
      'Continue engaging in activities that help you relax.',
      'Practice mindfulness or meditation.',
      'Spend time in nature.',
      'Listen to calming music.'
    ],
    happy: [
      'Keep doing activities that make you happy.',
      'Share your happiness with friends and family.',
      'Engage in hobbies you love.',
      'Stay active and exercise regularly.'
    ],
    fine: [
      'Maintain a balanced lifestyle.',
      'Ensure you have a good work-life balance.',
      'Stay connected with loved ones.',
      'Take breaks and relax when needed.'
    ],
    anxious: {
      studying: [
        'Practice deep breathing exercises.',
        'Prepare a study schedule to manage time effectively.',
        'List your tasks and prioritize them.',
        'Exercise or Move Around.',
        'Listen to Calming Music or Brown Noise'
      ],
      exam: [
        'Practice deep breathing exercises.',
        'Prepare a study schedule to manage time effectively.',
        'Take regular breaks during study sessions.',
        'Follow the Pomodoro Technique.',
        'Start Studying Early.',
        'List your tasks and prioritize them.',
        'Stay hydrated.'
      ],
      work: [
        'Practice deep breathing exercises.',
        'Take regular breaks during work.',
        'Talk to someone you trust about your feelings.',
        'Try relaxation techniques like yoga or meditation.'
      ],
      gaming: [
        'Set time limits for gaming sessions.',
        'Take breaks to relax and stretch.',
        'Engage in physical activities to reduce anxiety.',
        'Talk to someone you trust about your feelings.'
      ]
    },
    sad: {
      studying: [
        'Talk to a friend or family member about your feelings.',
        'Engage in activities that you enjoy.',
        'Consider seeking professional help if needed.',
        'Practice self-care and be kind to yourself.'
      ],
      exam: [
        'Talk to a friend or family member about your feelings.',
        'Engage in activities that you enjoy.',
        'Consider seeking professional help if needed.',
        'Practice self-care and be kind to yourself.'
      ],
      work: [
        'Talk to a friend or family member about your feelings.',
        'Engage in activities that you enjoy.',
        'Consider seeking professional help if needed.',
        'Practice self-care and be kind to yourself.'
      ],
      gaming: [
        'Talk to a friend or family member about your feelings.',
        'Engage in activities that you enjoy.',
        'Consider seeking professional help if needed.',
        'Practice self-care and be kind to yourself.'
      ]
    },
    angry: {
      studying: [
        'Practice deep breathing exercises to calm down.',
        'Engage in physical activities to release pent-up energy.',
        'Talk to someone you trust about your feelings.',
        'Try relaxation techniques like yoga or meditation.'
      ],
      exam: [
        'Practice deep breathing exercises to calm down.',
        'Engage in physical activities to release pent-up energy.',
        'Talk to someone you trust about your feelings.',
        'Try relaxation techniques like yoga or meditation.'
      ],
      work: [
        'Practice deep breathing exercises to calm down.',
        'Engage in physical activities to release pent-up energy.',
        'Talk to someone you trust about your feelings.',
        'Try relaxation techniques like yoga or meditation.'
      ],
      gaming: [
        'Practice deep breathing exercises to calm down.',
        'Engage in physical activities to release pent-up energy.',
        'Talk to someone you trust about your feelings.',
        'Try relaxation techniques like yoga or meditation.'
      ]
    }
  },
  sleepQualities: {
    'No Sleep': [
      'Establish a regular sleep schedule.',
      'Create a relaxing bedtime routine.',
      'Avoid caffeine and heavy meals before bed.',
      'Consider seeking professional help if needed.'
    ],
    'Poor': [
      'Maintain a consistent sleep schedule.',
      'Create a relaxing bedtime routine.',
      'Limit screen time before bed.',
      'Ensure your sleep environment is comfortable.'
    ],
    'Medium': [
      'Maintain a consistent sleep schedule.',
      'Create a relaxing bedtime routine.',
      'Avoid caffeine and heavy meals before bed.',
      'Ensure your sleep environment is comfortable.'
    ],
    'Good': [
      'Continue maintaining a consistent sleep schedule.',
      'Keep up with your relaxing bedtime routine.',
      'Ensure your sleep environment remains comfortable.',
      'Avoid caffeine and heavy meals before bed.'
    ]
  }
};

const shuffleArray = (array) => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
};

let cachedRecommendations = null;
let lastGeneratedDate = null;

const generateRecommendations = (correlationData) => {
  const recommendations = [];

  correlationData.forEach(result => {
    if (result.correlationMood && result.correlationActivity) {
      const mood = result.correlationMood.toLowerCase();
      const activity = result.correlationActivity.toLowerCase();

      if (recommendationsData.moods[mood] && recommendationsData.moods[mood][activity]) {
        const shuffledRecommendations = shuffleArray(recommendationsData.moods[mood][activity]);
        recommendations.push(...shuffledRecommendations.slice(0, 3));
      } else if (recommendationsData.moods[mood]) {
        const shuffledRecommendations = shuffleArray(recommendationsData.moods[mood]);
        recommendations.push(...shuffledRecommendations.slice(0, 3));
      }
    } else if (result.sleepQuality) {
      const sleepQuality = result.sleepQuality;

      if (recommendationsData.sleepQualities[sleepQuality]) {
        const shuffledRecommendations = shuffleArray(recommendationsData.sleepQualities[sleepQuality]);
        recommendations.push(...shuffledRecommendations.slice(0, 2));
      }
    }
  });

  return recommendations;
};

const getWeeklyCorrelationForCorrelation = async (req, res) => {
  try {
    const token = req.headers.authorization.split(' ')[1];
    if (!token) {
      return res.status(401).json({ message: 'No token found' });
    }

    const data = await MoodLog.find({ user: req.user.id });

    const startOfWeek = moment().startOf('isoWeek'); // Monday
    const endOfWeek = moment().endOf('isoWeek'); // Sunday
    const currentWeekLogs = data.filter(log => {
      const logDate = moment(log.date);
      return logDate.isBetween(startOfWeek, endOfWeek, null, '[]');
    });

    const moodActivityMap = {};
    const moodSocialMap = {};
    const sleepQualityCount = {
      'No Sleep': 0,
      'Poor Sleep': 0,
      'Medium Sleep': 0,
      'Good Sleep': 0
    };
    let healthCount = 0;

    currentWeekLogs.forEach(log => {
      const { mood, activities, social, health, sleepQuality } = log;

      // Handle activities
      activities.forEach(activity => {
        if (!moodActivityMap[mood]) {
          moodActivityMap[mood] = {};
        }

        if (!moodActivityMap[mood][activity]) {
          moodActivityMap[mood][activity] = 0;
        }

        moodActivityMap[mood][activity]++;
      });

      // Handle social
      social.forEach(social => {
        if (!moodSocialMap[mood]) {
          moodSocialMap[mood] = {};
        }

        if (!moodSocialMap[mood][social]) {
          moodSocialMap[mood][social] = 0;
        }

        moodSocialMap[mood][social]++;
      });

      // Handle health
      if (health.length > 0) {
        healthCount++;
      }

      // Handle sleep quality
      if (sleepQuality in sleepQualityCount) {
        sleepQualityCount[sleepQuality]++;
      }
    });

    const correlationResults = [];
    let topMood = null;
    let topMoodCount = 0;
    let topMoodActivity = null;
    let topMoodActivityCount = 0;

    Object.keys(moodActivityMap).forEach(mood => {
      const moodCount = Object.values(moodActivityMap[mood]).reduce((a, b) => a + b, 0);
      if (moodCount >= 3 && moodCount > topMoodCount) {
        topMood = mood;
        topMoodCount = moodCount;
        topMoodActivity = null;
        topMoodActivityCount = 0;

        Object.keys(moodActivityMap[mood]).forEach(activity => {
          if (moodActivityMap[mood][activity] > topMoodActivityCount) {
            topMoodActivity = activity;
            topMoodActivityCount = moodActivityMap[mood][activity];
          }
        });
      }
    });

    if (topMood && topMoodActivity) {
      const percentage = ((topMoodActivityCount / topMoodCount) * 100).toFixed(2);
      correlationResults.push({
        correlationValue: parseFloat(percentage),
        correlationMood: topMood,
        correlationActivity: topMoodActivity
      });
    }

    // Analyze social patterns
    let topSocialMood = null;
    let topSocialMoodCount = 0;
    let topSocialActivity = null;
    let topSocialActivityCount = 0;

    Object.keys(moodSocialMap).forEach(mood => {
      const moodCount = Object.values(moodSocialMap[mood]).reduce((a, b) => a + b, 0);
      if (moodCount >= 3 && moodCount > topSocialMoodCount) {
        topSocialMood = mood;
        topSocialMoodCount = moodCount;
        topSocialActivity = null;
        topSocialActivityCount = 0;

        Object.keys(moodSocialMap[mood]).forEach(social => {
          if (moodSocialMap[mood][social] > topSocialActivityCount) {
            topSocialActivity = social;
            topSocialActivityCount = moodSocialMap[mood][social];
          }
        });
      }
    });

    if (topSocialMood && topSocialActivity) {
      const percentage = ((topSocialActivityCount / topSocialMoodCount) * 100).toFixed(2);
      correlationResults.push({
        correlationValue: parseFloat(percentage),
        correlationMood: topSocialMood,
        correlationSocial: topSocialActivity
      });
    }

    // Analyze sleep quality patterns
    const totalSleepLogs = Object.values(sleepQualityCount).reduce((a, b) => a + b, 0);
    const poorSleepLogs = sleepQualityCount['No Sleep'] + sleepQualityCount['Poor Sleep'];
    const mediumSleepLogs = sleepQualityCount['Medium Sleep'];
    const goodSleepLogs = sleepQualityCount['Good Sleep'];

    const poorSleepPercentage = ((poorSleepLogs / totalSleepLogs) * 100).toFixed(2);
    const mediumSleepPercentage = ((mediumSleepLogs / totalSleepLogs) * 100).toFixed(2);
    const goodSleepPercentage = ((goodSleepLogs / totalSleepLogs) * 100).toFixed(2);

    const sleepQualityResults = [
      { quality: 'Poor', percentage: poorSleepPercentage, count: poorSleepLogs },
      { quality: 'Medium', percentage: mediumSleepPercentage, count: mediumSleepLogs },
      { quality: 'Good', percentage: goodSleepPercentage, count: goodSleepLogs }
    ];

    // Find the top sleep quality result
    const topSleepQuality = sleepQualityResults.reduce((prev, current) => (prev.count > current.count ? prev : current));
    correlationResults.push({
      sleepQualityValue: parseFloat(topSleepQuality.percentage),
      sleepQuality: topSleepQuality.quality
    });

    // Analyze health patterns
    correlationResults.push({
      healthStatus: healthCount > 3 ? 'normal' : 'insufficient'
    });

    // Generate recommendations
    const recommendations = generateRecommendations(correlationResults);

    // Check if today is Sunday and if the correlation results for the current week have already been stored
    const today = moment().day();
    if (today === 0) { // 0 represents Sunday
      const existingCorrelation = await CorrelationValue.findOne({
        user: req.user.id,
        createdAt: {
          $gte: new Date(startOfWeek),
          $lte: new Date(endOfWeek)
        }
      });

      if (!existingCorrelation) {
        const correlation = new CorrelationValue({
          user: req.user.id,
          correlationResults,
          recommendations
        });

        await correlation.save();
      }
    }

    res.status(200).json({ correlationResults, recommendations });
  } catch (error) {
    console.error('Error fetching data:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

const getWeeklyCorrelationForStatistics = async (req, res) => {
  try {
    const token = req.headers.authorization.split(' ')[1];
    if (!token) {
      return res.status(401).json({ message: 'No token found' });
    }

    const { startOfWeek, endOfWeek } = req.query;

    const data = await MoodLog.find({ user: req.user.id });

    const currentWeekLogs = data.filter(log => {
      const logDate = moment(log.date);
      return logDate.isBetween(startOfWeek, endOfWeek, null, '[]');
    });

    const moodActivityMap = {};
    const sleepQualityCount = {
      'No Sleep': 0,
      'Poor Sleep': 0,
      'Medium Sleep': 0,
      'Good Sleep': 0
    };

    currentWeekLogs.forEach(log => {
      const { mood, activities, sleepQuality } = log;
      activities.forEach(activity => {
        if (!moodActivityMap[mood]) {
          moodActivityMap[mood] = {};
        }

        if (!moodActivityMap[mood][activity]) {
          moodActivityMap[mood][activity] = 0;
        }

        moodActivityMap[mood][activity]++;
      });

      if (sleepQuality in sleepQualityCount) {
        sleepQualityCount[sleepQuality]++;
      }
    });

    const correlationResults = [];
    let topMood = null;
    let topMoodCount = 0;
    let topMoodActivity = null;
    let topMoodActivityCount = 0;

    Object.keys(moodActivityMap).forEach(mood => {
      const moodCount = Object.values(moodActivityMap[mood]).reduce((a, b) => a + b, 0);
      if (moodCount >= 3 && moodCount > topMoodCount) {
        topMood = mood;
        topMoodCount = moodCount;
        topMoodActivity = null;
        topMoodActivityCount = 0;

        Object.keys(moodActivityMap[mood]).forEach(activity => {
          if (moodActivityMap[mood][activity] > topMoodActivityCount) {
            topMoodActivity = activity;
            topMoodActivityCount = moodActivityMap[mood][activity];
          }
        });
      }
    });

    if (topMood && topMoodActivity) {
      const percentage = ((topMoodActivityCount / topMoodCount) * 100).toFixed(2);
      correlationResults.push({
        correlationValue: parseFloat(percentage),
        correlationMood: topMood,
        correlationActivity: topMoodActivity
      });
    }

    // Analyze sleep quality patterns
    const totalSleepLogs = Object.values(sleepQualityCount).reduce((a, b) => a + b, 0);
    const poorSleepLogs = sleepQualityCount['No Sleep'] + sleepQualityCount['Poor Sleep'];
    const mediumSleepLogs = sleepQualityCount['Medium Sleep'];
    const goodSleepLogs = sleepQualityCount['Good Sleep'];

    const poorSleepPercentage = totalSleepLogs > 0 ? ((poorSleepLogs / totalSleepLogs) * 100).toFixed(2) : 0;
    const mediumSleepPercentage = totalSleepLogs > 0 ? ((mediumSleepLogs / totalSleepLogs) * 100).toFixed(2) : 0;
    const goodSleepPercentage = totalSleepLogs > 0 ? ((goodSleepLogs / totalSleepLogs) * 100).toFixed(2) : 0;

    const sleepQualityResults = [
      { quality: 'Poor', percentage: poorSleepPercentage, count: poorSleepLogs },
      { quality: 'Medium', percentage: mediumSleepPercentage, count: mediumSleepLogs },
      { quality: 'Good', percentage: goodSleepPercentage, count: goodSleepLogs }
    ];

    // Find the top sleep quality result
    const topSleepQuality = sleepQualityResults.reduce((prev, current) => (prev.count > current.count ? prev : current));
    correlationResults.push({
      sleepQualityValue: parseFloat(topSleepQuality.percentage),
      sleepQuality: topSleepQuality.quality
    });

    // Removed code that adds correlation results to CorrelationValue schema

    res.status(200).json(correlationResults);
  } catch (error) {
    console.error('Error fetching data:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

const getWeeklyComparison = async (req, res) => {
  try {
    const token = req.headers.authorization.split(' ')[1];
    if (!token) {
      return res.status(401).json({ message: 'No token found' });
    }

    const { startOfWeek, endOfWeek } = req.query;

    const currentWeekCorrelation = await Correlation.findOne({
      user: req.user.id,
      createdAt: {
        $gte: new Date(startOfWeek),
        $lte: new Date(endOfWeek)
      }
    });

    const previousWeekCorrelation = await Correlation.findOne({
      user: req.user.id,
      createdAt: {
        $gte: moment(startOfWeek).subtract(1, 'weeks').toDate(),
        $lte: moment(endOfWeek).subtract(1, 'weeks').toDate()
      }
    });

    if (!currentWeekCorrelation || !previousWeekCorrelation) {
      return res.status(404).json({ message: 'Correlation data not found for comparison' });
    }

    const compareMoods = (current, previous) => {
      const moodValues = {
        relaxed: 5,
        happy: 4,
        fine: 3,
        anxious: 2,
        sad: 1,
        angry: 0
      };

      const currentMoodValue = moodValues[current.correlationMood.toLowerCase()] || 0;
      const previousMoodValue = moodValues[previous.correlationMood.toLowerCase()] || 0;

      const percentageChange = ((currentMoodValue - previousMoodValue) / 5) * 100;
      return percentageChange;
    };

    const currentMood = currentWeekCorrelation.correlationResults.find(result => result.correlationMood);
    const previousMood = previousWeekCorrelation.correlationResults.find(result => result.correlationMood);

    if (currentMood && previousMood) {
      const moodComparison = compareMoods(currentMood, previousMood);
      res.status(200).json({ moodComparison });
    } else {
      res.status(404).json({ message: 'Mood data not found for the specified weeks' });
    }
  } catch (error) {
    console.error('Error fetching comparison data:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

const getRecommendations = async (req, res) => {
  try {
    const today = moment().startOf('day');

    if (!cachedRecommendations || !lastGeneratedDate || !today.isSame(lastGeneratedDate, 'day')) {
      const { correlationData } = req.body;
      cachedRecommendations = generateRecommendations(correlationData);
      lastGeneratedDate = today;
    }

    res.status(200).json(cachedRecommendations);
  } catch (error) {
    console.error('Error generating recommendations:', error);
    res.status(500).json({ success: false, message: 'Server Error', error: error.message });
  }
};

module.exports = {
  getWeeklyCorrelationForCorrelation,
  getWeeklyCorrelationForStatistics,
  getWeeklyComparison,
  getRecommendations
};