# Leveraging Large Language Models for Wind Energy Assessment

Janice Anta Zebaze $^{1}$ , Azanzi Jiomekong $^{2,3}$ , Innocent Souopgui $^{4}$ , Germaine Djuidje Kenmoé $^{1}$

$^{1}$ Laboratory of Mechanics, Department of Physics, University of Yaounde I, Yaounde, Cameroon

$^{2}$ Department of Computer Science, University of Yaounde I, Yaounde, Cameroon

$^{3}$ TIB – Leibniz Information Centre for Science and Technology, Hannover, Germany

$^{4}$ RedLine Performance Solutions, LLC, USA

antazebaze@gmail.com, fidel.jiomekong@facsciences-uy1.cm, isouopgui@redlineperf.com, kdjuidje@yahoo.fr

1101 Pennsylvania Ave, NW Suite 300

Washington, DC 20004 USA

proceedings-questions@aaai.org

# Abstract

Economic growth and development require a consistent supply of energy. Energy which has mainly been supplied from fossil fuels. The impacts of these on the environment such as global warming, raised an alarm on their use. As a result other sources of energy such as wind energy are used as alternatives for electricity production. Wind energy assessment nevertheless faces barriers due to its stochastic nature. This later creates various regimes, which traditional models can't always fit thereby producing poor estimates. In this work, we aim to use Large Language Models (LLMs) to predict the wind potential in a given location. Through this approach, we aim at lifting the barrier on energy problems in developing countries by providing knowledge on the state of wind energy in given locations.

# Wind Energy Assessment

The growing world population along with technological advancements creates an increase in demand for energy. Energy required to run daily activities, households, industries, transports and enterprises. A consistent and /or sufficient supply in energy is therefore a sign of economic growth and development in a country. Nevertheless, developing countries still face energy supply issues. The amount of energy produced is not enough to meet the demand of the population, many regions are still off grid and frequent outages are still present. To meet up this demand in energy, fossil fuels have long been used worldwide. But their consequences on the environment such as global warming, raised an alarm on their use and the need for cleaner resources.

Among these resources, wind is a promising resource, whose implementation has been progressing these past years. Although its dominant presence in countries in S.E Asia, Europe and America, in Africa, its use is still shy. Its absence is noted in many countries in Africa. In Cameroon in particular, its implementation is still very scarce. It has as advantages its standalone capacity, enabling standalone supply in remote and grid off locations. Apart for electricity generation, it can be used to carry tasks such as water pumping and water desalination, which are of great use in developing countries.

Wind energy is a resource contained in the kinetic energy of wind. It is an intermittent and stochastic resource varying greatly in time and space. Present worldwide at different intensities, its potential therefore varies from one location to another, and increasingly with height. As a result, every wind project is preceded by an assessment phase. During which one evaluates the amount of energy that can be extracted from a given location. This assessment validates the feasibility and economic viability of such projects.

# Research Problem

Assessing wind energy potential starts by modeling the wind speeds. Statistical distributions such as Weibull, Gamma, Nakagami and the Normal distributions are used for this end (Lencastre, Yazidi, and Lind 2024). But wind due its stochastic nature has various regimes and patterns. This renders the task difficult since one distribution may model well a regime in one location and fail in another location. Hence, they are not suitable for all locations, rendering the assessment task difficult. Among these distributions, the Weibull distribution has long been used and proven successful in most studies. But although its success, it also face limits in locations with dominant low speed regimes (Chang 2011), reflecting its modeling limits. To cope with this, AI methods such as artificial neural networks are used for used (Fousseni, Kodjo, and Adjanoh 2021). Their introduction into the field provide predictive features over stations without measurements. By taking in more parameters they provide an overall good result. The main limit of such methods reside on the choice of the adequate parameters of the model such as the number of neurons, hidden layers or optimization method.

In our work, we aim at using Large Language models to assess the wind potential in a given area. Large Language models are trending AI tool used and seen to be successful in several domains such as food information engineering (Jiomekong et al. 2023) and healthcare (Thirunavukarasu et al. 2023). Nevertheless, their presence in the field of wind energy assessment is still to be observed. The use of LLMs for wind energy potential assessment is therefore a novelty we bring to this field through our work. By using LLMs such as BERT, we aim to assess the wind energy potential predict the wind potential a given location in Cameroon. The use

of LLMs was also pushed after an extensive literature review, during which scientific papers were selected, data extracted, organized and documented. All this was saved onto the Open Research Knowledge Graph (ORKG) for open access. Through this study the different wind speed distributions models studied in different papers were extracted and compared $^{1}$ . From this study, one could notice the absence of LLMs in wind energy potential assessment. Through this work, we aim at making available knowledge on the potential of wind energy in a given area, with a particular attention on developing countries. This could boost their economy, by increasing power supply, creating skilled jobs, developing the industrial sector and achieving several sustainable development goals such as affordable and clean energy and decent work and economic growth.

# Research Objectives

Our primary objective is to use LLMs for wind energy assessment. This latter can then be broken down as follows:

- Predicting the wind speed probabilities and power densities using LLMs.   
• Wind speed extrapolation.   
- Predicting the available energy at the heights 10m, 50m and 100m.

# Research Methodology

Data Collection: To realize these objectives, data was collected in the Western region of Cameroon, at a frequency of 10 minutes at the height of 10m. The collected raw data was pre-processed such that it can be used by the LLM. From here, i evaluated the wind speed values at the heights of 50m and 100m using the Power law. The resulting dataset is available on hugging face $^{2}$ .

Statistical Modeling: The wind speed probabilities were fitted using the Weibull distribution given as:

$$
f _ {W} (V; \alpha , k) = \frac {k}{\alpha} \left(\frac {V}{\alpha}\right) ^ {k - 1} \exp \left[ - \left(\frac {V}{\alpha}\right) ^ {k} \right] \tag {1}
$$

where $\alpha$ in $(m / s)$ , is the scale parameter and $k$ is the dimensionless shape parameter. These parameters where evaluated using the Energy Pattern Factor method.

Prediction using LLMs: To predict the wind probabilities using BERT, 80% of the data will be used for training and the remaining 20% for testing. To predict the wind speed probabilities, the model will take as input the wind speeds and their frequencies. Next, the wind potential and energy density shall be predicted at the heights of 10m, 50m and 100m using the values of the wind speed at these heights respectively. Along this, the wind speed will be predicted at 10m height, taking as input parameters the wind speed, the temperature, direction humidity and date and at the output the wind speed. At other heights the inputs shall be the values of these wind speeds at the respective heights.

Evaluation The performance of these tests shall be evaluated using the F1-score, the precision and recall to evaluate the performance of the LLM. Also, the correlation coefficient, the mean absolute error shall be used to compare the performance of the the LLM with the different theoretical models.

Expected results are modeling performances better than the famous Weibull distribution. Better error metrics than previous works done in the domain. Thanks to this work, the available energy at different heights in one area can be predicted on one hand. On the other hand, one can then be able to know if a site can be exploited for energy production and the amount of energy which can be produced orienting the choice of a wind turbine. Its main advantage is on its expected adaptability to any wind regime over other distributions. Breaking the main limits of these distributions and providing insights on their use in other fields where these distributions are used.

# Conclusion

The use of LLMs is promising and gaining terrain over several domains. Their consequent use in this field can be a breakthrough to energy issues in developing countries. That by providing detailed information on the wind characteristics of a given location and therefore suitability of a site for energy production. This information itself is the first step for any project on wind assessment and will therefore serve as a motivation in areas which are still unaware of their wind energy potential. Also, in the scientific community, the modeling abilities of LLMs over commonly used distributions will be a breakthrough. Their flexibility to all wind types fueled by the amount number of parameters they take into account which is much greater than regular distributions.

# Acknowledgments

I thank the TIB - Leibniz Information Centre for Science and Technology for the support they provided during the ORKG curation grant 2024.

# References

Chang, T. P. 2011. Estimation of wind energy potential using different probability density functions. Applied energy, 88(5): 1848–1856.   
Fousseni, A.; Kodjo, M. K.; and Adjanoh, A. A. 2021. Static and dynamic evaluation of wind potential in the Kara region of Togo using artificial neural networks. TH Wildau Engineering and Natural Sciences Proceedings, 1.   
Jiomekong, A.; Oelen, A.; Auer, S.; Anna-Lena, L.; and Lars, V. 2023. Food information engineering. AI Magazine. Lencastre, P.; Yazidi, A.; and Lind, P. G. 2024. Modeling Wind-Speed Statistics beyond the Weibull Distribution. Energies, 17(11): 2621.   
Thirunavukarasu, A. J.; Ting, D. S. J.; Elangovan, K.; Gutierrez, L.; Tan, T. F.; and Ting, D. S. W. 2023. Large language models in medicine. Nature medicine, 29(8): 1930–1940.