$(document).ready(function() {
    console.log('NBA Player Analogy Finder initialized');
    
    // Cache DOM elements
    const $loading = $('.loading-spinner');
    const $error = $('.error-message');
    const $analogyResults = $('#analogyResults');
    const $runAnalogyBtn = $('#runAnalogy');
    const $searchButtonText = $('#searchButtonText');
    
    // Initialize Select2 for player selectors
    $('.player-select').select2({
        placeholder: 'Type to search for a player...',
        allowClear: true,
        width: '100%',
        theme: 'bootstrap-5',
        language: {
            noResults: function() {
                return 'No players found. Try a different search term.';
            },
            inputTooShort: function(args) {
                return `Type ${args.minimum - args.input.length} more characters to search`;
            },
            searching: function() {
                return 'Searching...';
            },
            errorLoading: function() {
                return 'Error loading results. Please try again.';
            }
        },
        ajax: {
            url: '/search_players',
            dataType: 'json',
            delay: 250,
            data: function(params) {
                const playerId = $(this).attr('id');
                const yearSelector = playerId === 'pA' ? '#yearA' : '#yearB';
                
                return {
                    q: params.term || '',
                    year: $(yearSelector).val() || ''
                };
            },
            processResults: function(data) {
                return {
                    results: data.map(function(player) {
                        const baseName = player.replace(/\s*\(\d{4}-\d{2}\)$/, '').trim();
                        const year = player.match(/\((\d{4}-\d{2})\)$/)?.[1] || '';
                        
                        return { 
                            id: player, 
                            text: player,
                            baseName: baseName,
                            year: year
                        };
                    })
                };
            },
            cache: true
        },
        minimumInputLength: 2,
        templateResult: formatPlayerResult,
        templateSelection: formatPlayerSelection,
        escapeMarkup: function(markup) {
            return markup;
        }
    });

    // Format how each result appears in the dropdown
    function formatPlayerResult(player) {
        if (!player.id) return player.text;
        
        const $result = $(`
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <div class="fw-bold">${escapeHtml(player.baseName)}</div>
                    ${player.year ? `<small class="text-muted">${player.year} Season</small>` : ''}
                </div>
            </div>
        `);
        
        return $result;
    }

    // Format how the selected item appears
    function formatPlayerSelection(player) {
        return player.baseName || player.text;
    }

    // Update labels when players are selected
    $('#pA').on('select2:select', function(e) {
        const playerName = e.params.data.baseName || e.params.data.text.split('(')[0].trim();
        $('#labelA, #labelA2').text(playerName);
        updateSearchButtonText();
    });

    $('#pB').on('select2:select', function(e) {
        const playerName = e.params.data.baseName || e.params.data.text.split('(')[0].trim();
        $('#labelB, #labelB2').text(playerName);
        updateSearchButtonText();
    });

    // Clear labels when players are cleared
    $('#pA').on('select2:clear', function() {
        $('#labelA, #labelA2').text('A');
        updateSearchButtonText();
    });

    $('#pB').on('select2:clear', function() {
        $('#labelB, #labelB2').text('B');
        updateSearchButtonText();
    });

    // Handle direction toggle
    $('input[name="analogyDirection"]').on('change', function() {
        updateSearchButtonText();
    });

    // Update search button text based on current selection
    function updateSearchButtonText() {
        const playerA = $('#pA').select2('data')[0]?.baseName || 'A';
        const playerB = $('#pB').select2('data')[0]?.baseName || 'B';
        const direction = $('input[name="analogyDirection"]:checked').val();
        
        if (direction === 'a-b') {
            $searchButtonText.text(`Find "${playerB} of ${playerA}"`);
        } else {
            $searchButtonText.text(`Find "${playerA} of ${playerB}"`);
        }
        
        // Also update the radio button labels
        $('#labelA, #labelA2').text(playerA);
        $('#labelB, #labelB2').text(playerB);
    }

    // Handle year filter changes (refresh player search)
    $('#yearA, #yearB').on('change', function() {
        const playerId = $(this).attr('id') === 'yearA' ? '#pA' : '#pB';
        $(playerId).val(null).trigger('change');
    });

    // Handle analogy button click
    $runAnalogyBtn.on('click', runAnalogy);

    function runAnalogy() {
        const playerA = $('#pA').val();
        const playerB = $('#pB').val();
        const yearA = $('#yearA').val();
        const yearB = $('#yearB').val();
        const topK = parseInt($('#analogyTop').val()) || 5;
        const direction = $('input[name="analogyDirection"]:checked').val();
        const excludeInputPlayers = $('#excludeInputPlayers').is(':checked');
        const excludeSameTeam = $('#excludeSameTeam').is(':checked');
        
        console.log('Run analogy clicked', { playerA, playerB, yearA, yearB, topK, direction, excludeInputPlayers, excludeSameTeam });
        
        if (!playerA || !playerB) {
            showError('Please select both players for the analogy');
            return;
        }
        
        if (playerA === playerB && yearA === yearB) {
            showError('Please select different players or different seasons');
            return;
        }
        
        // Show loading state
        $analogyResults.hide();
        $loading.removeClass('d-none');
        $error.empty().addClass('d-none');
        $runAnalogyBtn.prop('disabled', true);
        
        // Make API request
        $.ajax({
            url: '/analogy',
            method: 'GET',
            data: {
                a: playerA,
                b: playerB,
                year_a: yearA,
                year_b: yearB,
                direction: direction,
                top: topK,
                exclude_input_players: excludeInputPlayers,
                exclude_same_team: excludeSameTeam
            },
            dataType: 'json',
            success: function(data) {
                console.log('Analogy API Response:', data);
                if (data.error) {
                    showError(data.error);
                } else {
                    displayAnalogyResults(data);
                }
            },
            error: function(xhr, status, error) {
                console.error('Analogy API Error:', { status, error, response: xhr.responseText });
                let errorMsg = 'Error running analogy';
                try {
                    const response = JSON.parse(xhr.responseText);
                    errorMsg = response.error || errorMsg;
                } catch (e) {
                    errorMsg += ': ' + (xhr.statusText || error);
                }
                showError(errorMsg);
            },
            complete: function() {
                $loading.addClass('d-none');
                $runAnalogyBtn.prop('disabled', false);
            }
        });
    }
    
    function formatScore(score) {
        return Math.round(score * 1000) / 10; // Convert to percentage with 1 decimal
    }

    function getScoreColor(score) {
        // Convert score to 0-100 scale
        const percent = score * 100;
        if (percent > 80) return '#28a745'; // Green
        if (percent > 60) return '#5cb85c'; // Light green
        if (percent > 40) return '#ffc107'; // Yellow
        if (percent > 20) return '#fd7e14'; // Orange
        return '#dc3545'; // Red
    }

    function getPositionBadge(player) {
        if (!player.position) return '';
        
        const { primary, is_guard, is_forward, is_center } = player.position;
        let badgeClass = 'bg-secondary';
        
        if (is_guard) badgeClass = 'bg-primary';
        else if (is_forward) badgeClass = 'bg-success';
        else if (is_center) badgeClass = 'bg-danger';
        
        return `<span class="badge ${badgeClass} position-badge" title="Primary Position">${primary || 'N/A'}</span>`;
    }

    function getPhysicalInfo(player) {
        if (!player.physical) return '';
        
        const { height, weight, bmi } = player.physical;
        if (!height && !weight) return '';
        
        const heightStr = height ? `${Math.floor(height/12)}'${Math.round(height%12)}\"` : 'N/A';
        const weightStr = weight ? `${weight} lbs` : 'N/A';
        const bmiStr = bmi ? bmi.toFixed(1) : 'N/A';
        
        return `
            <div class="physical-info">
                <small class="text-muted d-block">
                    <i class="bi bi-rulers"></i> ${heightStr} • ${weightStr}
                </small>
                <small class="text-muted">
                    <i class="bi bi-graph-up"></i> BMI: ${bmiStr}
                </small>
            </div>`;
    }

    function getScoreTooltip(scores, weights) {
        if (!scores) return '';
        
        let tooltip = '<div class="score-tooltip">';
        tooltip += '<div class="score-breakdown">';
        
        // Add weighted scores
        tooltip += '<div class="score-section"><strong>Score Components</strong><br>';
        
        // Sort components by weighted score (highest first)
        const sortedScores = Object.entries(scores).sort((a, b) => b[1].weighted - a[1].weighted);
        
        sortedScores.forEach(([key, scoreData]) => {
            const weight = weights ? (weights[key] || 0) * 100 : 0;
            const weighted = formatScore(scoreData.weighted);
            const raw = formatScore(scoreData.raw);
            
            tooltip += `
                <div class="score-row">
                    <span class="score-label">${key.replace('_', ' ').toUpperCase()}:</span>
                    <span class="score-value">${weighted}%</span>
                    <div class="progress" style="height: 4px;">
                        <div class="progress-bar" 
                             role="progressbar" 
                             style="width: ${weighted}%; background-color: ${getScoreColor(scoreData.weighted)}"
                             aria-valuenow="${weighted}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                    <small class="text-muted">
                        Raw: ${raw}% • Weight: ${weight.toFixed(0)}%
                    </small>
                </div>`;
        });
        
        tooltip += '</div>'; // Close score-section
        tooltip += '</div>'; // Close score-breakdown
        tooltip += '</div>'; // Close score-tooltip
        
        return tooltip;
    }

    function displayAnalogyResults(data) {
        console.log('Displaying analogy results:', data);
        $analogyResults.empty();
        
        if (!data || !data.results || data.results.length === 0) {
            const noResultsMsg = 'No analogy results found. Try different players or seasons.';
            $analogyResults.html(`
                <div class="alert alert-info">
                    <i class="bi bi-info-circle me-2"></i>${noResultsMsg}
                </div>
            `);
            $analogyResults.show();
            return;
        }

        // Get the base names without years
        const getBaseName = (name) => name ? name.split(' (')[0].trim() : '';
        
        const playerA = data.a || 'Player A';
        const playerB = data.b || 'Player B';
        const baseA = getBaseName(playerA);
        const baseB = getBaseName(playerB);
        
        // Get the direction from the URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        const direction = urlParams.get('direction') || 'a-b';
        
        // Format the query based on direction
        let query;
        if (direction === 'a-b') {
            query = `${baseA} of ${baseB}`;
        } else {
            query = `${baseB} of ${baseA}`;
        }

        let html = `
            <div class="card shadow border-0">
                <div class="card-header bg-primary text-white">
                    <div class="text-center">
                        <p class="mb-0 opacity-75">Results based on statistical vector analysis</p>
                    </div>
                </div>
                <div class="card-body p-4">
                    <div class="alert alert-info">
                        <i class="bi bi-info-circle me-2"></i>
                        Hover over match scores to see detailed breakdown of similarity components.
                    </div>
                    <div class="row g-4">
        `;

        data.results.forEach((player, index) => {
            const similarity = Math.round(player.similarity * 100);
            const playerYear = player.season || '';
            const playerName = player.name.replace(/\s*\(\d{4}-\d{2}\)$/, '').trim();
            
            // Add animation delay for staggered effect
            const animationDelay = index * 100;
            
            // Determine badge color based on ranking
            let badgeClass = 'bg-success';
            if (index === 0) badgeClass = 'bg-warning text-dark';
            else if (index === 1) badgeClass = 'bg-info';
            else if (index === 2) badgeClass = 'bg-secondary';
            
            // Get position badge and physical info
            const positionBadge = getPositionBadge(player);
            const physicalInfo = getPhysicalInfo(player);
            
            // Generate tooltip for scores
            const scoreTooltip = getScoreTooltip(player.scores, data.score_weights);
            
            html += `
                <div class="col-md-6 col-lg-4" style="opacity: 0; animation: fadeIn 0.5s ease-out ${animationDelay}ms forwards;">
                    <div class="card h-100 border-0 shadow-sm hover-lift">
                        <div class="card-body text-center">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <span class="badge ${badgeClass} rounded-pill fs-6 px-3 py-1">#${index + 1}</span>
                                ${positionBadge}
                            </div>
                            
                            <h5 class="card-title mb-1">${escapeHtml(playerName)}</h5>
                            ${playerYear ? `<h6 class="card-subtitle text-muted mb-2">${playerYear} Season</h6>` : '<div class="mb-2"></div>'}
                            
                            ${physicalInfo}
                            
                            <div class="mt-3 mb-3 position-relative" data-bs-toggle="tooltip" data-bs-html="true" 
                                 title="${scoreTooltip.replace(/"/g, '&quot;')}">
                                <div class="display-4 fw-bold" style="color: ${getScoreColor(player.similarity)}; cursor: help;">
                                    ${similarity}%
                                </div>
                                <small class="text-muted">Match Score <i class="bi bi-info-circle"></i></small>
                            </div>
                            
                            <div class="progress" style="height: 10px;">
                                <div class="progress-bar" 
                                     role="progressbar" 
                                     style="width: ${similarity}%; background-color: ${getScoreColor(player.similarity)};" 
                                     aria-valuenow="${similarity}" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        html += `
                    </div>
                </div>
            </div>
        `;
        
        $analogyResults.html(html).show();
    }
    
    // Helper function to escape HTML
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Helper function to show error messages
    function showError(message) {
        console.error('Error:', message);
        $error.html(`
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="bi bi-exclamation-triangle-fill me-2"></i>
                <strong>Error:</strong> ${escapeHtml(message)}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `).removeClass('d-none');
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            $error.find('.alert').alert('close');
        }, 5000);
    }
});
