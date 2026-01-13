# Projet d'analyse de la marchabilité

Ce dossier contient les notebooks et les fichiers associés pour l'analyse de la marchabilité des parcours urbains, incluant à la fois la synchronisation des vidéos et l'analyse des notes et feedbacks des participants.

## Structure du dossier

- `synchronisation_segments_participants.ipynb`  
  Notebook dédié à la **synchronisation des vidéos**.  
  - Compare et synchronise les vidéos enregistréess.  
  - Utilise la métrique SSIM pour identifier les correspondances entre images des vidéos.  
  - Implémente un filtrage par médiane glissante pour stabiliser la détection des pics.  
  - Permet d'identifier les points de correspondance pour faciliter l'analyse vidéo et les études ultérieures sur le mouvement et le trafic.

- `analyse_retours_qualitatifs_et_quantitatifs.ipynb`  
  Notebook pour l’**analyse des notes et des feedbacks des participants**.  
  - Prépare les données de notation par sous-segment.  
  - Analyse quantitative des notes (moyennes, médianes, boxplots).  
  - Analyse qualitative des feedbacks :  
    - Extraction des points à partir des phrases.  
    - Clustering par sous-segment (Agglomerative Clustering).  
    - Clustering global (HDBSCAN) pour regrouper les thèmes communs.  
    - Analyse des corrélations entre thèmes et notes des sous-segments.