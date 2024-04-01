library(readxl)
library(tidyverse)
library(stargazer)

################################################################################
# Create data
################################################################################

# Load datasets
results <- read_excel("Data/Raw/NFL Schedule.xlsx")
status <- read_excel("Data/Raw/NFL Player Status.xlsx")

# Create injury data at team-week-season level
out <- status %>%
  group_by(Team, Season) %>%
  mutate(age = mean(Age_Start_Season)) %>%
  ungroup() %>%
  mutate(out = ifelse(Active_Inactive == "Out", 1, 0)) %>%
  group_by(Team, Season, Week, age) %>%
  summarise(out = sum(out)) %>%
  arrange(Team, Season, Week)

# Create a win loss data at team-week-season level
winners <- results %>%
  select(Week, Season, Team = `Winner/tie`) %>%
  mutate(win = 1)
losers <- results %>%
  select(Week, Season, Team = `Loser/tie`) %>%
  mutate(win = 0)
win_loss <- bind_rows(winners, losers) %>% arrange(Team, Season, Week)

# Merge injuries & average age to win_loss
finaldata <- merge(win_loss, out, by = c("Team", "Season", "Week"), all.x = TRUE)

# Account for Relocated teams
finaldata <- finaldata %>%
  mutate(Team = ifelse(Team == "Washington Redskins", "Washington Football Team", Team)) %>%
  mutate(Team = ifelse(Team == "St. Louis Rams", "Los Angeles Rams", Team)) %>%
  mutate(Team = ifelse(Team == "San Diego Chargers", "Los Angeles Chargers", Team))

# Create win percentage by team-season & merge it to finaldata
team_season_wins <- finaldata %>% 
  mutate(Team = ifelse(Team == "Washington Redskins", "Washington Football Team", Team)) %>%
  mutate(Team = ifelse(Team == "St. Louis Rams", "Los Angeles Rams", Team)) %>%
  mutate(Team = ifelse(Team == "San Diego Chargers", "Los Angeles Chargers", Team)) %>%
  group_by(Team, Season) %>% 
  summarise(win_perc = mean(win)) %>% 
  mutate(lag_win_perc = lag(win_perc))
finaldata <- merge(finaldata, team_season_wins, by = c("Team", "Season"))

# Lowercase varnames
names(finaldata) <- tolower(names(finaldata)) 

# Filter out 2015
finaldata = filter(finaldata, season != 2015)

# Save data
save(finaldata, file = "final_data.rda")

################################################################################
# Preliminary Analysis
################################################################################

load("final_data.rda")

# Summary statistics
finaldata %>%
  select(win, out, age, lag_win_perc) %>% 
  as.data.frame() %>% 
  stargazer(type = 'text', digits = 2, median=TRUE)

# Relationship between wins & injuries
tab <- finaldata %>% 
  group_by(win) %>% 
  summarise(out)
ggplot(tab, aes(y=out, x=as.factor(win)))+
  geom_bar(stat="identity") +
  theme_classic()

# Correlation table
finaldata %>% 
  select(win, out, age, lag_win_perc) %>% 
  cor() %>% 
  stargazer(type = 'text', digits = 2)

################################################################################
# Regression Analysis
################################################################################

finaldata <- finaldata
lm_model1 <- lm(win ~ out, finaldata)
lm_model2 <- lm(win ~ out + age, finaldata)
lm_model3 <- lm(win ~ out + age + lag_win_perc, finaldata)
lm_model4a <- lm(win ~ out + as.factor(team), finaldata)
lm_model5a <- lm(win ~ out + as.factor(team) + age + lag_win_perc, finaldata)
lm_model4b <- lm(win ~ out + as.factor(team) + as.factor(season), finaldata)
lm_model5b <- lm(win ~ out + as.factor(team) + as.factor(season) + age + lag_win_perc, finaldata)

stargazer(lm_model1, lm_model2, lm_model3, lm_model4a, lm_model5b, type="text",
          keep = c("out", "age", "lag_win_perc"))

finaldata <- finaldata
glm_model1 <- glm(win ~ out, family = binomial, finaldata)
glm_model2 <- glm(win ~ out + age, family = binomial, finaldata)
glm_model3 <- glm(win ~ out + age + lag_win_perc, family = binomial, finaldata)
glm_model4a <- glm(win ~ out + as.factor(team), family = binomial, finaldata)
glm_model5a <- glm(win ~ out + as.factor(team) + age + lag_win_perc, family = binomial, finaldata)
glm_model4b <- glm(win ~ out + as.factor(team) + as.factor(season), family = binomial, finaldata)
glm_model5b <- glm(win ~ out + as.factor(team) + as.factor(season) + age + lag_win_perc, family = binomial, finaldata)

stargazer(glm_model1,glm_model2, glm_model3, glm_model4a, glm_model5b, type="text",
          keep = c("out", "age", "lag_win_perc"))

################################################################################